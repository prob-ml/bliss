import torch

from bliss.catalog import BaseTileCatalog
from bliss.encoder.convnet import CatalogNet, ContextNet
from bliss.encoder.encoder import Encoder
from case_studies.galaxy_clustering.encoder.convnet import GalaxyClusterFeaturesNet


class GalaxyClusterEncoder(Encoder):
    def __init__(self, *args, downsample_at_front: bool = True, **kwargs):
        self.downsample_at_front = downsample_at_front
        super().__init__(*args, **kwargs)

    def initialize_networks(self):
        """Load the convolutional neural networks that map normalized images to catalog parameters.
        This method can be overridden to use different network architectures.
        `checkerboard_net` and `second_net` can be left as None if not needed.
        """
        power_of_two = (self.tile_slen != 0) & (self.tile_slen & (self.tile_slen - 1) == 0)
        assert power_of_two, "tile_slen must be a power of two"
        ch_per_band = self.image_normalizer.num_channels_per_band()
        num_features = 256
        self.features_net = GalaxyClusterFeaturesNet(
            len(self.image_normalizer.bands),
            ch_per_band,
            num_features,
            tile_slen=self.tile_slen,
            downsample_at_front=self.downsample_at_front,
        )
        n_params_per_source = self.var_dist.n_params_per_source
        self.marginal_net = CatalogNet(num_features, n_params_per_source)
        self.checkerboard_net = ContextNet(num_features, n_params_per_source)
        if self.double_detect:
            self.second_net = CatalogNet(num_features, n_params_per_source)

    def get_features_and_parameters(self, batch):
        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)
        x_cat_marginal = self.marginal_net(x_features)
        return x_features, x_cat_marginal

    def sample(self, batch, use_mode=True):
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]

        x_features, x_cat_marginal = self.get_features_and_parameters(batch)

        marginal_cat = self.var_dist.sample(x_cat_marginal, use_mode=use_mode)

        if not self.use_checkerboard:
            est_cat = marginal_cat
        else:
            ht, wt = h // self.tile_slen, w // self.tile_slen
            tile_cb = self._get_checkerboard(ht, wt).squeeze(1)
            white_history_mask = tile_cb.expand([batch_size, -1, -1])

            white_context = self.make_context(marginal_cat, white_history_mask)
            x_cat_white = self.checkerboard_net(x_features, white_context)
            white_cat = self.var_dist.sample(x_cat_white, use_mode=use_mode)
            est_cat = self.interleave_catalogs(marginal_cat, white_cat, white_history_mask)

        if self.double_detect:
            x_cat_second = self.second_net(x_features)
            second_cat = self.var_dist.sample(x_cat_second, use_mode=use_mode)
            # our loss function implies that the second detection is ignored for a tile
            # if the first detection is empty for that tile
            second_cat["n_sources"] *= est_cat["n_sources"]
            est_cat = est_cat.union(second_cat)

        return est_cat.symmetric_crop(self.tiles_to_crop)

    def update_metrics(self, batch, batch_idx):
        target_cat = BaseTileCatalog(self.tile_slen, batch["tile_catalog"])
        target_cat = target_cat.symmetric_crop(self.tiles_to_crop)

        mode_cat = self.sample(batch, use_mode=True)
        self.mode_metrics.update(target_cat, mode_cat)

        sample_cat = self.sample(batch, use_mode=False)
        self.sample_metrics.update(target_cat, sample_cat)

    def on_validation_epoch_end(self):
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.report_metrics(self.sample_metrics, "val/sample", show_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Pytorch lightning method."""
        with torch.no_grad():
            return {
                "mode_cat": self.sample(batch, use_mode=True),
                # we may want multiple samples
                "sample_cat": self.sample(batch, use_mode=False),
                "parameters": self.get_features_and_parameters(batch)[1],
            }
