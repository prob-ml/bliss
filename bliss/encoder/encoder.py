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
from bliss.encoder.data_augmentation import augment_batch
from bliss.encoder.image_normalizer import ImageNormalizer
from bliss.encoder.metrics import CatalogMatcher
from bliss.encoder.plots import PlotCollection
from bliss.encoder.variational_dist import VariationalDistSpec


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
        vd_spec: VariationalDistSpec,
        metrics: MetricCollection,
        plots: PlotCollection,
        matcher: CatalogMatcher,
        min_flux_threshold: float = 0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        do_data_augmentation: bool = False,
        compile_model: bool = False,
        double_detect: bool = False,
        use_checkerboard: bool = True,
    ):
        """Initializes Encoder.

        Args:
            survey_bands: all band-pass filters available for this survey
            tile_slen: dimension in pixels of a square tile
            tiles_to_crop: margin of tiles not to use for computing loss
            image_normalizer: object that applies input transforms to images
            vd_spec: object that makes a variational distribution from raw convnet output
            plots: for plotting relevant images and outputs during training
            metrics: for scoring predicted catalogs during training
            matcher: for matching predicted catalogs to ground truth catalogs
            min_flux_threshold: Sources with a lower flux will not be considered when computing loss
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
            do_data_augmentation: used for determining whether or not do data augmentation
            compile_model: compile model for potential performance improvements
            double_detect: whether to make up to two detections per tile rather than one
            use_checkerboard: whether to use dependent tiling
        """
        super().__init__()

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.tiles_to_crop = tiles_to_crop
        self.image_normalizer = image_normalizer
        self.vd_spec = vd_spec
        self.mode_metrics = metrics.clone()
        self.sample_metrics = metrics.clone()
        self.plots = plots
        self.matcher = matcher
        self.min_flux_threshold = min_flux_threshold
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.do_data_augmentation = do_data_augmentation
        self.double_detect = double_detect
        self.use_checkerboard = use_checkerboard

        ch_per_band = self.image_normalizer.num_channels_per_band()
        assert tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        num_features = 256
        self.features_net = FeaturesNet(
            len(image_normalizer.bands),
            ch_per_band,
            num_features,
            double_downsample=(tile_slen == 4),
        )
        n_params_per_source = vd_spec.n_params_per_source
        self.marginal_net = CatalogNet(num_features, n_params_per_source)
        self.checkerboard_net = ContextNet(num_features, n_params_per_source)
        if self.double_detect:
            self.second_net = CatalogNet(num_features, n_params_per_source)

        if compile_model:
            self.features_net = torch.compile(self.features_net)
            self.marginal_net = torch.compile(self.marginal_net)
            self.checkerboard_net = torch.compile(self.checkerboard_net)
            if self.double_detect:
                self.second_net = torch.compile(self.second_net)

    def _get_checkerboard(self, ht, wt):
        # make/store a checkerboard of tiles
        # https://stackoverflow.com/questions/72874737/how-to-make-a-checkerboard-in-pytorch
        arange_ht = torch.arange(ht, device=self.device)
        arange_wt = torch.arange(wt, device=self.device)
        mg = torch.meshgrid(arange_ht, arange_wt, indexing="ij")
        indices = torch.stack(mg)
        tile_cb = indices.sum(axis=0) % 2
        return rearrange(tile_cb, "ht wt -> 1 1 ht wt")

    def infer_conditional(self, x_features, history_cat, history_mask):
        masked_cat = copy(history_cat)
        # masks not just n_sources; n_sources controls access to all fields.
        # does not mutate history_cat because we aren't using *=
        masked_cat.n_sources = masked_cat.n_sources * history_mask

        # we may want to use richer conditioning information in the future;
        # e.g., a residual image based on the catalog so far
        detection_history = masked_cat.n_sources > 0

        context = torch.stack([detection_history, history_mask], dim=1).float()
        x_cat = self.checkerboard_net(x_features, context)
        return self.vd_spec.make_dist(x_cat)

    def interleave_catalogs(self, marginal_cat, cond_cat, marginal_mask):
        d = {}
        mm5d = rearrange(marginal_mask, "b ht wt -> b ht wt 1 1")
        for k, v in marginal_cat.to_dict().items():
            mm = marginal_mask if k == "n_sources" else mm5d
            d[k] = v * mm + cond_cat[k] * (1 - mm)
        return TileCatalog(self.tile_slen, d)

    def infer(self, batch, history_callback):
        batch_size = batch["images"].size(0)
        h, w = batch["images"].shape[2:4]
        ht, wt = h // self.tile_slen, w // self.tile_slen
        pred = {}

        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)

        x_cat_marginal = self.marginal_net(x_features)
        x_features = x_features.detach()  # is this helpful? doing it here to match old code
        pred["x_features"] = x_features
        pred["marginal"] = self.vd_spec.make_dist(x_cat_marginal)

        history_cat = history_callback(pred["marginal"])
        pred["history_cat"] = history_cat

        if self.use_checkerboard:
            tile_cb = self._get_checkerboard(ht, wt).squeeze(1)
            white_history_mask = tile_cb.expand([batch_size, -1, -1])
            pred["white_history_mask"] = white_history_mask

            pred["white"] = self.infer_conditional(x_features, history_cat, white_history_mask)
            pred["black"] = self.infer_conditional(x_features, history_cat, 1 - white_history_mask)

        if self.double_detect:
            x_cat_second = self.second_net(x_features)
            pred["second"] = self.vd_spec.make_dist(x_cat_second)

        return pred

    def sample(self, batch, use_mode=True):
        def marginal_detections(pred_marginal):  # noqa: WPS430
            return pred_marginal.sample(use_mode=use_mode)

        pred = self.infer(batch, marginal_detections)

        if not self.use_checkerboard:
            est_cat = pred["history_cat"]
        else:
            white_cat = pred["white"].sample(use_mode=use_mode)
            est_cat = self.interleave_catalogs(
                pred["history_cat"], white_cat, pred["white_history_mask"]
            )
        if self.double_detect:
            est_cat_s = pred["second"].sample(use_mode=use_mode)
            # our loss function implies that the second detection is ignored for a tile
            # if the first detection is empty for that tile
            est_cat_s.n_sources *= est_cat.n_sources
            est_cat = est_cat.union(est_cat_s)
        return est_cat.symmetric_crop(self.tiles_to_crop)

    def _single_detection_nll(self, target_cat, pred):
        marginal_loss = pred["marginal"].compute_nll(target_cat)

        if not self.use_checkerboard:
            return marginal_loss

        white_loss = pred["white"].compute_nll(target_cat)
        white_loss_mask = 1 - pred["white_history_mask"]
        white_loss *= white_loss_mask

        black_loss = pred["black"].compute_nll(target_cat)
        black_loss_mask = pred["white_history_mask"]
        black_loss *= black_loss_mask

        # we divide by two because we score two predictions for each tile
        return (marginal_loss + white_loss + black_loss) / 2

    def _double_detection_nll(self, target_cat1, target_cat, pred):
        target_cat2 = target_cat.get_brightest_sources_per_tile(band=2, exclude_num=1)

        nll_marginal_z1 = self._single_detection_nll(target_cat1, pred)
        nll_cond_z2 = pred["second"].compute_nll(target_cat2)
        nll_marginal_z2 = self._single_detection_nll(target_cat2, pred)
        nll_cond_z1 = pred["second"].compute_nll(target_cat1)

        none_mask = target_cat.n_sources == 0
        loss0 = nll_marginal_z1 * none_mask

        one_mask = target_cat.n_sources == 1
        loss1 = (nll_marginal_z1 + nll_cond_z2) * one_mask

        two_mask = target_cat.n_sources >= 2
        loss2a = nll_marginal_z1 + nll_cond_z2
        loss2b = nll_marginal_z2 + nll_cond_z1
        lse_stack = torch.stack([loss2a, loss2b], dim=-1)
        loss2_unmasked = -torch.logsumexp(-lse_stack, dim=-1)
        loss2 = loss2_unmasked * two_mask

        return loss0 + loss1 + loss2

    def _compute_loss(self, batch, logging_name):
        batch_size = batch["images"].size(0)
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        # filter out undetectable sources
        target_cat = target_cat.filter_tile_catalog_by_flux(min_flux=self.min_flux_threshold)

        # make predictions/inferences
        target_cat1 = target_cat.get_brightest_sources_per_tile(band=2, exclude_num=0)
        truth_callback = lambda _: target_cat1
        pred = self.infer(batch, truth_callback)

        # compute loss
        if not self.double_detect:
            loss = self._single_detection_nll(target_cat1, pred)
        else:
            loss = self._double_detection_nll(target_cat1, target_cat, pred)

        # exclude border tiles and report average per-tile loss
        ttc = self.tiles_to_crop
        interior_loss = pad(loss, [-ttc, -ttc, -ttc, -ttc])
        loss = interior_loss.sum() / interior_loss.numel()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        if self.do_data_augmentation:
            augment_batch(batch)

        return self._compute_loss(batch, "train")

    def update_metrics(self, batch):
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])
        target_cat = target_cat.filter_tile_catalog_by_flux(min_flux=self.min_flux_threshold)
        target_cat = target_cat.symmetric_crop(self.tiles_to_crop).to_full_catalog()

        mode_cat = self.sample(batch, use_mode=True).to_full_catalog()
        matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.mode_metrics.update(target_cat, mode_cat, matching)

        sample_cat = self.sample(batch, use_mode=False).to_full_catalog()
        matching = self.matcher.match_catalogs(target_cat, sample_cat)
        self.sample_metrics.update(target_cat, sample_cat, matching)

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "val")
        self.update_metrics(batch)

        plot_dict = {
            "restrict_batch": 0,
            "batch": batch,
            "batch_idx": batch_idx,
            "max_iterations": self.trainer.max_epochs,
            "tile_slen": self.tile_slen,
            "min_flux_threshold": self.min_flux_threshold,
            "tiles_to_crop": self.tiles_to_crop,
            "sample": self.sample,
            "current_iteration": self.current_epoch,
            "logging_name": "val",
        }
        self.plots.plot_all(state_dict=plot_dict, check_freqs=True, logger=self.logger)

    def report_metrics(self, metrics, logging_name, show_epoch=False):
        for k, v in metrics.compute().items():
            self.log(f"{logging_name}/{k}", v, sync_dist=True)

        for metric_name, metric in metrics.items():
            if hasattr(metric, "plot"):  # noqa: WPS421
                fig, _axes = metric.plot()
                name = f"Epoch:{self.current_epoch}" if show_epoch else ""
                name += f"/{logging_name} {metric_name}"
                if self.logger:
                    self.logger.experiment.add_figure(name, fig)

        metrics.reset()

    def on_validation_epoch_end(self):
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.report_metrics(self.sample_metrics, "val/sample", show_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "test")
        self.update_metrics(batch)

    def on_test_epoch_end(self):
        self.report_metrics(self.mode_metrics, "test/mode", show_epoch=False)
        self.report_metrics(self.sample_metrics, "test/sample", show_epoch=False)

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
