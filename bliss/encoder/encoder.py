from copy import copy
from typing import Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection

from bliss.catalog import TileCatalog
from bliss.encoder.convnet import CatalogNet, ContextNet, FeaturesNet
from bliss.encoder.image_normalizer import ImageNormalizer
from bliss.encoder.metrics import CatalogMatcher
from bliss.encoder.variational_dist import VariationalDist
from bliss.global_env import GlobalEnv


class Encoder(pl.LightningModule):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

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
        min_flux_for_loss: float = 0,
        min_flux_for_metrics: float = 0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        compile_model: bool = False,
        double_detect: bool = False,
        use_checkerboard: bool = True,
        reference_band: int = 2,
    ):
        """Initializes Encoder.

        Args:
            survey_bands: all band-pass filters available for this survey
            tile_slen: dimension in pixels of a square tile
            tiles_to_crop: margin of tiles not to use for computing loss
            image_normalizer: object that applies input transforms to images
            var_dist: object that makes a variational distribution from raw convnet output
            sample_image_renders: for plotting relevant images (overlays, shear maps)
            metrics: for scoring predicted catalogs during training
            matcher: for matching predicted catalogs to ground truth catalogs
            min_flux_for_loss: Sources with a lower flux will not be considered when computing loss
            min_flux_for_metrics: filter sources by flux during test
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
            compile_model: compile model for potential performance improvements
            double_detect: whether to make up to two detections per tile rather than one
            use_checkerboard: whether to use dependent tiling
            reference_band: band to use for filtering sources
        """
        super().__init__()

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.tiles_to_crop = tiles_to_crop
        self.image_normalizer = image_normalizer
        self.var_dist = var_dist
        self.metrics = metrics
        self.sample_image_renders = sample_image_renders
        self.matcher = matcher
        self.min_flux_for_loss = min_flux_for_loss
        self.min_flux_for_metrics = min_flux_for_metrics
        assert self.min_flux_for_loss <= self.min_flux_for_metrics, "invalid threshold"
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.compile_model = compile_model
        self.double_detect = double_detect
        self.use_checkerboard = use_checkerboard
        self.reference_band = reference_band

        self.initialize_networks()

        if self.compile_model:
            self.features_net = torch.compile(self.features_net)
            self.marginal_net = torch.compile(self.marginal_net)
            if self.use_checkerboard:
                self.checkerboard_net = torch.compile(self.checkerboard_net)
            if self.double_detect:
                self.second_net = torch.compile(self.second_net)

    def initialize_networks(self):
        """Load the convolutional neural networks that map normalized images to catalog parameters.
        This method can be overridden to use different network architectures.
        `checkerboard_net` and `second_net` can be left as None if not needed.
        """
        assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        ch_per_band = self.image_normalizer.num_channels_per_band()
        num_features = 256
        self.features_net = FeaturesNet(
            len(self.image_normalizer.bands),
            ch_per_band,
            num_features,
            double_downsample=(self.tile_slen == 4),
        )
        n_params_per_source = self.var_dist.n_params_per_source
        self.marginal_net = CatalogNet(num_features, n_params_per_source)
        self.checkerboard_net = ContextNet(num_features, n_params_per_source)
        if self.double_detect:
            self.second_net = CatalogNet(num_features, n_params_per_source)

    def _get_checkerboard(self, ht, wt):
        # make/store a checkerboard of tiles
        # https://stackoverflow.com/questions/72874737/how-to-make-a-checkerboard-in-pytorch
        arange_ht = torch.arange(ht, device=self.device)
        arange_wt = torch.arange(wt, device=self.device)
        mg = torch.meshgrid(arange_ht, arange_wt, indexing="ij")
        indices = torch.stack(mg)
        tile_cb = indices.sum(axis=0) % 2
        return rearrange(tile_cb, "ht wt -> 1 1 ht wt")

    def make_context(self, history_cat, history_mask):
        masked_cat = copy(history_cat)
        # masks not just n_sources; n_sources controls access to all fields.
        # does not mutate history_cat because we aren't using *=
        masked_cat["n_sources"] = masked_cat["n_sources"] * history_mask

        # we may want to use richer conditioning information in the future;
        # e.g., a residual image based on the catalog so far
        detection_history = masked_cat["n_sources"] > 0

        return torch.stack([detection_history, history_mask], dim=1).float()

    def interleave_catalogs(self, marginal_cat, cond_cat, marginal_mask):
        d = {}
        mm5d = rearrange(marginal_mask, "b ht wt -> b ht wt 1 1")
        for k, v in marginal_cat.items():
            mm = marginal_mask if k == "n_sources" else mm5d
            d[k] = v * mm + cond_cat[k] * (1 - mm)
        return TileCatalog(self.tile_slen, d)

    def sample(self, batch, use_mode=True):
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]

        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)

        x_cat_marginal = self.marginal_net(x_features)
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

    def _double_detection_nll(self, target_cat1, target_cat, pred):
        target_cat2 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=1
        )

        nll_marginal_z1 = self._single_detection_nll(target_cat1, pred)
        nll_cond_z2 = self.var_dist.compute_nll(pred["x_cat_second"], target_cat2)
        nll_marginal_z2 = self._single_detection_nll(target_cat2, pred)
        nll_cond_z1 = self.var_dist.compute_nll(pred["x_cat_second"], target_cat1)

        none_mask = target_cat["n_sources"] == 0
        loss0 = nll_marginal_z1 * none_mask

        one_mask = target_cat["n_sources"] == 1
        loss1 = (nll_marginal_z1 + nll_cond_z2) * one_mask

        two_mask = target_cat["n_sources"] >= 2
        loss2a = nll_marginal_z1 + nll_cond_z2
        loss2b = nll_marginal_z2 + nll_cond_z1
        lse_stack = torch.stack([loss2a, loss2b], dim=-1)
        loss2_unmasked = -torch.logsumexp(-lse_stack, dim=-1)
        loss2 = loss2_unmasked * two_mask

        return loss0 + loss1 + loss2

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
        if not self.double_detect:
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

    def on_fit_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def on_train_epoch_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        return self._compute_loss(batch, "train")

    def update_metrics(self, batch, batch_idx):
        target_tile_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])
        target_tile_cat = target_tile_cat.filter_by_flux(
            min_flux=self.min_flux_for_metrics,
            band=self.reference_band,
        )
        target_cat = target_tile_cat.symmetric_crop(self.tiles_to_crop).to_full_catalog()

        mode_tile_cat = self.sample(batch, use_mode=True).filter_by_flux(
            min_flux=self.min_flux_for_metrics
        )
        mode_cat = mode_tile_cat.to_full_catalog()
        matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.metrics.update(target_cat, mode_cat, matching)

        self.sample_image_renders.update(
            batch,
            target_cat,
            mode_tile_cat,
            mode_cat,
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
        self.report_metrics(self.metrics, "val/mode", show_epoch=True)
        self.report_metrics(self.sample_image_renders, "val/image_renders", show_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "test")
        self.update_metrics(batch, batch_idx)

    def on_test_epoch_end(self):
        self.report_metrics(self.metrics, "test/mode", show_epoch=False)

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
