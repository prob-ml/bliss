import itertools
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
from bliss.encoder.convnet import ContextNet, FeaturesNet
from bliss.encoder.image_normalizer import ImageNormalizer
from bliss.encoder.metrics import CatalogMatcher
from bliss.encoder.variational_dist import VariationalDist


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
        self.mode_metrics = metrics.clone()
        self.sample_metrics = metrics.clone()
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

        # Generate all binary combinations for n^2 elements
        n = 2
        binary_combinations = list(itertools.product([0, 1], repeat=n * n))
        mask_patterns = torch.tensor(binary_combinations).view(-1, n, n)  # noqa: WPS114
        self.register_buffer("mask_patterns", mask_patterns)

        self.initialize_networks()

        if self.compile_model:
            self.features_net = torch.compile(self.features_net)
            self.context_net = torch.compile(self.context_net)

    def initialize_networks(self):
        """Load the convolutional neural networks that map normalized images to catalog parameters.
        This method can be overridden to use different network architectures.
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
        self.context_net = ContextNet(num_features, n_params_per_source)

    def _get_checkerboard(self, ht, wt):
        # make/store a checkerboard of tiles
        # https://stackoverflow.com/questions/72874737/how-to-make-a-checkerboard-in-pytorch
        arange_ht = torch.arange(ht, device=self.device)
        arange_wt = torch.arange(wt, device=self.device)
        mg = torch.meshgrid(arange_ht, arange_wt, indexing="ij")
        indices = torch.stack(mg)
        tile_cb = indices.sum(axis=0) % 2
        return rearrange(tile_cb, "ht wt -> 1 1 ht wt")

    def make_context(self, history_cat, history_mask, detection2=False):
        if history_cat is None:
            detection_history = torch.zeros_like(history_mask)
        else:
            masked_cat = copy(history_cat)
            # masks not just n_sources; n_sources controls access to all fields.
            # does not mutate history_cat because we aren't using *=

            masked_cat["n_sources"] = masked_cat["n_sources"] * history_mask
            # we may want to use richer conditioning information in the future;
            # e.g., a residual image based on the catalog so far
            detection_history = masked_cat["n_sources"] > 0

        id_func = torch.ones_like if detection2 else torch.zeros_like
        detection_id = id_func(detection_history)

        return torch.stack([detection_history, history_mask, detection_id], dim=1).float()

    def interleave_catalogs(self, marginal_cat, cond_cat, marginal_mask):
        d = {}
        mm5d = rearrange(marginal_mask, "b ht wt -> b ht wt 1 1")
        for k, v in marginal_cat.items():
            mm = marginal_mask if k == "n_sources" else mm5d
            d[k] = v * mm + cond_cat[k] * (1 - mm)
        return TileCatalog(self.tile_slen, d)

    def sample(self, batch, use_mode=True):
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]
        ht, wt = h // self.tile_slen, w // self.tile_slen

        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)

        est_cat1 = None
        for mask_pattern in self.mask_patterns[(0, 8, 12, 14), ...]:
            mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
            context1 = self.make_context(est_cat1, mask)
            x_cat1 = self.context_net(x_features, context1)
            new_est_cat = self.var_dist.sample(x_cat1, use_mode=use_mode)
            new_est_cat["n_sources"] *= 1 - mask
            if est_cat1 is None:
                est_cat1 = new_est_cat
            else:
                est_cat1["n_sources"] *= mask
                est_cat1 = est_cat1.union(new_est_cat, disjoint=True)

            if not self.use_checkerboard:
                break

        if self.double_detect:
            no_mask = torch.ones_like(mask)
            # could add some context from target_cat2 here, masked by `mask`
            context2 = self.make_context(est_cat1, no_mask, detection2=True)
            x_cat2 = self.context_net(x_features, context2)
            est_cat2 = self.var_dist.sample(x_cat2, use_mode=use_mode)
            # our loss function implies that the second detection is ignored for a tile
            # if the first detection is empty for that tile
            est_cat2["n_sources"] *= est_cat1["n_sources"]
            est_cat = est_cat1.union(est_cat2, disjoint=False)

        return est_cat.symmetric_crop(self.tiles_to_crop)

    def _compute_loss(self, batch, logging_name):
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]
        ht, wt = h // self.tile_slen, w // self.tile_slen

        # filter out undetectable sources and split catalog by flux
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])
        target_cat = target_cat.filter_by_flux(
            min_flux=self.min_flux_for_loss,
            band=self.reference_band,
        )
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat2 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=1
        )

        # new loss calculation
        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)

        loss = torch.zeros_like(x_features[:, 0, :, :])

        # could use all the mask patterns but memory is tight and these 4 are the ones
        # we actually use for sampling
        for mask_pattern in self.mask_patterns[(0, 8, 12, 14), ...]:
            mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
            context1 = self.make_context(target_cat1, mask)
            x_cat1 = self.context_net(x_features, context1)
            loss11 = self.var_dist.compute_nll(x_cat1, target_cat1)

            # could upweight some patterns that are under-represented or limit loss to
            # the quarter of tiles that are actually used for sampling
            loss += loss11 * (1 - mask)

            if not self.use_checkerboard:
                break

        if self.double_detect:
            no_mask = torch.ones_like(mask)
            context2 = self.make_context(target_cat1, no_mask, detection2=True)
            x_cat2 = self.context_net(x_features, context2)
            loss22 = self.var_dist.compute_nll(x_cat2, target_cat2)
            loss += loss22

        # exclude border tiles and report average per-tile loss
        ttc = self.tiles_to_crop
        interior_loss = pad(loss, [-ttc, -ttc, -ttc, -ttc])
        # could normalize by the number of tile predictions, rather than number of tiles
        loss = interior_loss.sum() / interior_loss.numel()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)

        return loss

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
            min_flux=self.min_flux_for_metrics,
            band=self.reference_band,
        )
        mode_cat = mode_tile_cat.to_full_catalog()
        matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.mode_metrics.update(target_cat, mode_cat, matching)

        sample_cat = self.sample(batch, use_mode=False)
        sample_cat = sample_cat.filter_by_flux(
            min_flux=self.min_flux_for_metrics,
            band=self.reference_band,
        )
        sample_cat = sample_cat.to_full_catalog()

        matching = self.matcher.match_catalogs(target_cat, sample_cat)
        self.sample_metrics.update(target_cat, sample_cat, matching)

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
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.report_metrics(self.sample_metrics, "val/sample", show_epoch=True)
        self.report_metrics(self.sample_image_renders, "val/image_renders", show_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "test")
        self.update_metrics(batch, batch_idx)

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
