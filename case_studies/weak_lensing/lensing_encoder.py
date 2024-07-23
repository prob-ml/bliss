from typing import Optional

import torch
from torchmetrics import MetricCollection

from bliss.catalog import BaseTileCatalog
from bliss.encoder.encoder import Encoder
from bliss.encoder.variational_dist import VariationalDist
from case_studies.weak_lensing.lensing_convnet import WeakLensingCatalogNet, WeakLensingFeaturesNet


class WeakLensingEncoder(Encoder):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        image_normalizers: list,
        var_dist: VariationalDist,
        sample_image_renders: MetricCollection,
        mode_metrics: MetricCollection,
        sample_metrics: Optional[MetricCollection] = None,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        reference_band: int = 2,
        **kwargs,
    ):
        super().__init__(
            survey_bands=survey_bands,
            tile_slen=tile_slen,
            image_normalizers=image_normalizers,
            var_dist=var_dist,
            matcher=None,
            sample_image_renders=sample_image_renders,
            mode_metrics=mode_metrics,
            sample_metrics=sample_metrics,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            use_double_detect=False,
            use_checkerboard=False,
            reference_band=reference_band,
        )

        self.initialize_networks()

    # override
    def initialize_networks(self):
        num_features = 256
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.features_net = WeakLensingFeaturesNet(
            n_bands=len(self.survey_bands),
            ch_per_band=ch_per_band,
            num_features=num_features,
            tile_slen=self.tile_slen,
        )
        self.catalog_net = WeakLensingCatalogNet(
            in_channels=num_features,
            out_channels=self.var_dist.n_params_per_source,
        )

    def sample(self, batch, use_mode=True):
        # multiple image normalizers
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        inputs = torch.cat(input_lst, dim=2)

        x_features = self.features_net(inputs)
        x_cat_marginal = self.catalog_net(x_features)
        # est cat
        return self.var_dist.sample(x_cat_marginal, use_mode=use_mode, return_base_cat=True)

    def _compute_loss(self, batch, logging_name):
        batch_size, _, _, _ = batch["images"].shape[0:4]

        target_cat = BaseTileCatalog(batch["tile_catalog"])

        # multiple image normalizers
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        inputs = torch.cat(input_lst, dim=2)

        pred = {}
        x_features = self.features_net(inputs)
        pred["x_cat_marginal"] = self.catalog_net(x_features)
        loss = self.var_dist.compute_nll(pred["x_cat_marginal"], target_cat)

        loss = loss.sum() / loss.numel()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)

        return loss

    def update_metrics(self, batch, batch_idx):
        target_cat = BaseTileCatalog(batch["tile_catalog"])

        mode_cat = self.sample(batch, use_mode=True)
        self.mode_metrics.update(target_cat, mode_cat, None)

        sample_cat_no_mode = self.sample(batch, use_mode=False)
        self.sample_metrics.update(target_cat, sample_cat_no_mode, None)

        self.sample_image_renders.update(
            batch,
            target_cat,
            mode_cat,
            None,
            self.current_epoch,
            batch_idx,
        )
