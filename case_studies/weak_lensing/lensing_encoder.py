import itertools
import sys
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection

from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder
from bliss.encoder.image_normalizer import ImageNormalizer
from bliss.encoder.metrics import CatalogMatcher
from bliss.encoder.variational_dist import VariationalDist
from bliss.global_env import GlobalEnv
from case_studies.weak_lensing.lensing_convnet import CatalogNet, FeaturesNet


class WeakLensingEncoder(Encoder):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        tiles_to_crop: int,
        image_normalizer: ImageNormalizer,
        var_dist: VariationalDist,
        metrics: MetricCollection,
        sample_image_renders: MetricCollection,
        matcher: CatalogMatcher,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        use_checkerboard: bool = False,
        reference_band: int = 2,
        **kwargs,
    ):
        super().__init__(
            survey_bands,
            tile_slen,
            tiles_to_crop,
            image_normalizer,
            var_dist,
            metrics,
            sample_image_renders,
            matcher,
            -sys.maxsize - 1,
            -sys.maxsize - 1,
            optimizer_params,
            scheduler_params,
            False,
            use_checkerboard,
            reference_band,
        )

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.tiles_to_crop = tiles_to_crop
        self.image_normalizer = image_normalizer
        self.var_dist = var_dist
        self.mode_metrics = metrics.clone()
        self.sample_metrics = metrics.clone()
        self.sample_image_renders = sample_image_renders
        self.matcher = matcher
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.use_checkerboard = use_checkerboard
        self.reference_band = reference_band

        self.initialize_networks()

    # override
    def initialize_networks(self):
        # assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"

        num_features = 256

        self.features_net = FeaturesNet(
            n_bands=len(self.image_normalizer.bands),
            ch_per_band=self.image_normalizer.num_channels_per_band(),
            num_features=num_features,
            tile_slen=self.tile_slen,
        )
        self.catalog_net = CatalogNet(
            num_features=num_features * 3,
            out_channels=self.var_dist.n_params_per_source,
        )

    def make_context(self, history_cat, history_mask):
        return torch.zeros((1, 6, 1, 1))  # TODO: make dynamic based on slen

    def sample(self, batch, use_mode=True):
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]
        ht, wt = h // self.tile_slen, w // self.tile_slen

        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)
        mask = torch.zeros([batch_size, ht, wt])
        context = self.make_context(None, mask).to("cuda")
        x_cat_marginal = self.catalog_net(x_features, context)
        est_cat = self.var_dist.sample(x_cat_marginal, use_mode=use_mode)
        return est_cat.symmetric_crop(self.tiles_to_crop)

    def _compute_loss(self, batch, logging_name):
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]
        ht, wt = h // self.tile_slen, w // self.tile_slen

        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        pred = {}

        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)
        mask = torch.zeros([batch_size, ht, wt])
        context = self.make_context(None, mask).to("cuda")
        pred["x_cat_marginal"] = self.catalog_net(x_features, context)

        # todo: rewrite loss function as shear mse + conv mse
        loss = self.var_dist.compute_nll(pred["x_cat_marginal"], target_cat)

        # exclude border tiles and report average per-tile loss
        ttc = self.tiles_to_crop
        interior_loss = pad(loss, [-ttc, -ttc, -ttc, -ttc])
        # could normalize by the number of tile predictions, rather than number of tiles
        loss = interior_loss.sum() / interior_loss.numel()
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
        target_tile_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        target_cat = target_tile_cat.symmetric_crop(self.tiles_to_crop).to_full_catalog()

        sample_tile_cat = self.sample(batch, use_mode=True)
        sample_cat = sample_tile_cat.to_full_catalog()
        # matching = self.matcher.match_catalogs(target_cat, sample_cat)
        self.mode_metrics.update(target_cat, sample_cat, None)

        sample_tile_cat = self.sample(batch, use_mode=False)
        sample_cat = sample_tile_cat.to_full_catalog()
        # smatching = self.matcher.match_catalogs(target_cat, sample_cat)
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
