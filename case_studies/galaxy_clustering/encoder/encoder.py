import torch

from bliss.catalog import BaseTileCatalog, TileCatalog
from bliss.encoder.encoder import Encoder
from case_studies.galaxy_clustering.encoder.convnet import (
    GalaxyClusterCatalogNet,
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
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        num_features = 256
        self.net = GalaxyClusterFeaturesNet(
            len(self.survey_bands),
            ch_per_band,
            num_features,  # output dimension
            tile_slen=self.tile_slen,
            downsample_at_front=self.downsample_at_front,
        )
        self.catalog_net = GalaxyClusterCatalogNet(num_features, self.var_dist.n_params_per_source)

    def make_context(self, history_cat, history_mask, detection2=False):
        detection_id = (
            torch.ones_like(history_mask) if detection2 else torch.zeros_like(history_mask)
        ).unsqueeze(1)
        if history_cat is None:
            # detections (1), flux (1), and locs (2) are the properties we condition on
            masked_history = (
                torch.zeros_like(history_mask, dtype=torch.float).unsqueeze(1).expand(-1, 4, -1, -1)
            )
        else:
            centered_locs = history_cat["locs"][..., 0, :] - 0.5
            log_fluxes = (history_cat.on_nmgy.squeeze(3).sum(-1) + 1).log()
            history_encoding_lst = [
                history_cat["n_sources"].float(),  # detection history
                log_fluxes * history_cat["n_sources"],  # flux history
                centered_locs[..., 0] * history_cat["n_sources"],  # x history
                centered_locs[..., 1] * history_cat["n_sources"],  # y history
            ]
            masked_history_lst = [v * history_mask for v in history_encoding_lst]
            masked_history = torch.stack(masked_history_lst, dim=1)

        return torch.concat([detection_id, history_mask.unsqueeze(1), masked_history], dim=1)

    def get_features_and_parameters(self, batch):
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        x = torch.cat(input_lst, dim=2)
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]
        ht, wt = h // self.tile_slen, w // self.tile_slen
        x_features = self.net(x)
        mask = torch.zeros([batch_size, ht, wt])
        context = self.make_context(None, mask).to("cuda")
        x_cat_marginal = self.catalog_net(x_features, context)
        return x_features, x_cat_marginal

    def sample(self, batch, use_mode=True):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.sample(x_cat_marginal, use_mode=use_mode)

    def update_metrics(self, batch, batch_idx):
        target_cat = BaseTileCatalog(batch["tile_catalog"])

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
                "parameters": self.get_parameters(batch)[1],
            }

    def _compute_loss(self, batch, logging_name):
        batch_size = batch["images"].shape[0]
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        loss = self.var_dist.compute_nll(x_cat_marginal, target_cat1)
        loss = loss.sum() / loss.numel()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)
        return loss
