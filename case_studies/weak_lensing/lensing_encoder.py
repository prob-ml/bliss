import sys
from typing import Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection

from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder
from bliss.encoder.variational_dist import VariationalDist
from bliss.global_env import GlobalEnv
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
            min_flux_for_loss=-sys.maxsize - 1,
            min_flux_for_metrics=-sys.maxsize - 1,
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
        return self.var_dist.sample(x_cat_marginal, use_mode=use_mode)

    def _compute_loss(self, batch, logging_name):
        batch_size, _, _, _ = batch["images"].shape[0:4]

        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

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

    def on_fit_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def on_train_epoch_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        return self._compute_loss(batch, "train")

    def update_metrics(self, batch, batch_idx):
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        sample_tile_cat = self.sample(batch, use_mode=True)
        sample_cat = sample_tile_cat.to_full_catalog()
        self.mode_metrics.update(target_cat, sample_cat, None)

        sample_tile_cat = self.sample(batch, use_mode=False)
        sample_cat = sample_tile_cat.to_full_catalog()
        self.sample_metrics.update(target_cat, sample_cat, None)

        self.sample_image_renders.update(
            batch,
            target_cat,
            sample_tile_cat,
            sample_cat,
            self.current_epoch,
            batch_idx,
        )

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "val")
        self.update_metrics(batch, batch_idx)

    def report_metrics(self, metrics, logging_name, show_epoch=False):
        for k, v in metrics.compute().items():
            self.log(f"{logging_name}/{k}", v, sync_dist=True)

        for metric_name, metric in metrics.items():
            if hasattr(metric, "plot"):  # noqa: WPS421
                plot_or_none = metric.plot()
                name = f"Epoch:{self.current_epoch}" if show_epoch else ""
                name += f"/{logging_name} {metric_name}"
                if self.logger and plot_or_none:
                    fig, _axes = plot_or_none
                    self.logger.experiment.add_figure(name, fig)

        metrics.reset()

    def on_validation_epoch_end(self):
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.report_metrics(self.sample_metrics, "sample/mode", show_epoch=True)
        self.report_metrics(self.sample_image_renders, "val/image_renders", show_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "test")
        self.update_metrics(batch, batch_idx)

    def on_test_epoch_end(self):
        self.report_metrics(self.mode_metrics, "test/mode", show_epoch=False)
        self.report_metrics(self.sample_metrics, "test/mode", show_epoch=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Pytorch lightning method."""
        with torch.no_grad():
            return {
                "mode_cat": self.sample(batch, use_mode=True),
                # we may want multiple samples
                "sample_cat": self.sample(batch, use_mode=False),
            }

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]
