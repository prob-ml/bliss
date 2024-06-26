import torch
from torch.nn.functional import pad

from bliss.catalog import BaseTileCatalog, TileCatalog
from bliss.encoder.encoder import Encoder
from case_studies.galaxy_clustering.encoder.convnet import (
    GalaxyClusterCatalogNet,
    GalaxyClusterContextNet,
    GalaxyClusterFeaturesNet,
)


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
        self.marginal_net = GalaxyClusterCatalogNet(num_features, n_params_per_source)
        self.checkerboard_net = GalaxyClusterContextNet(num_features, n_params_per_source)
        if self.use_double_detect:
            self.second_net = GalaxyClusterCatalogNet(num_features, n_params_per_source)

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

        if self.use_double_detect:
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

    def _compute_loss(self, batch, logging_name):
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]

        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        # filter out undetectable sources
        target_cat = target_cat.filter_by_flux(
            min_flux=self.min_flux_for_loss,
            band=self.reference_band,
        )
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )

        # make predictions/inferences
        pred = {}

        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)

        pred["x_cat_marginal"] = self.marginal_net(x_features)
        x_features = x_features.detach()  # is this helpful? doing it here to match old code

        if self.use_checkerboard:
            ht, wt = h // self.tile_slen, w // self.tile_slen
            tile_cb = self._get_checkerboard(ht, wt).squeeze(1)
            white_history_mask = tile_cb.expand([batch_size, -1, -1])
            pred["white_history_mask"] = white_history_mask

            white_context = self.make_context(target_cat1, white_history_mask)
            pred["x_cat_white"] = self.checkerboard_net(x_features, white_context)

            black_context = self.make_context(target_cat1, 1 - white_history_mask)
            pred["x_cat_black"] = self.checkerboard_net(x_features, black_context)

        # compute loss
        if not self.use_double_detect:
            loss = self._single_detection_nll(target_cat1, pred)
        else:
            pred["x_cat_second"] = self.second_net(x_features)
            loss = self._double_detection_nll(target_cat1, target_cat, pred)

        # exclude border tiles and report average per-tile loss
        ttc = self.tiles_to_crop
        interior_loss = pad(loss, [-ttc, -ttc, -ttc, -ttc])
        loss = interior_loss.sum() / interior_loss.numel()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)

        return loss

    def _single_detection_nll(self, target_cat, pred):
        marginal_loss = self.var_dist.compute_nll(pred["x_cat_marginal"], target_cat)

        if not self.use_checkerboard:
            return marginal_loss

        white_loss = self.var_dist.compute_nll(pred["x_cat_white"], target_cat)
        white_loss_mask = 1 - pred["white_history_mask"]
        white_loss *= white_loss_mask

        black_loss = self.var_dist.compute_nll(pred["x_cat_black"], target_cat)
        black_loss_mask = pred["white_history_mask"]
        black_loss *= black_loss_mask

        # we divide by two because we score two predictions for each tile
        return (marginal_loss + white_loss + black_loss) / 2
