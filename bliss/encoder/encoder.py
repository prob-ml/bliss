import itertools
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection

from bliss.catalog import TileCatalog
from bliss.encoder.convnets import CatalogNet, FeaturesNet
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
        image_normalizer: ImageNormalizer,
        var_dist: VariationalDist,
        matcher: CatalogMatcher,
        sample_image_renders: MetricCollection,
        mode_metrics: MetricCollection,
        sample_metrics: Optional[MetricCollection] = None,
        min_flux_for_loss: float = 0,
        min_flux_for_metrics: float = 0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        use_double_detect: bool = False,
        use_checkerboard: bool = True,
        reference_band: int = 2,
    ):
        """Initializes Encoder.

        Args:
            survey_bands: all band-pass filters available for this survey
            tile_slen: dimension in pixels of a square tile
            image_normalizer: object that applies input transforms to images
            var_dist: object that makes a variational distribution from raw convnet output
            matcher: for matching predicted catalogs to ground truth catalogs
            sample_image_renders: for plotting relevant images (overlays, shear maps)
            mode_metrics: for scoring predicted mode catalogs during training
            sample_metrics: for scoring predicted sampled catalogs during training
            min_flux_for_loss: Sources with a lower flux will not be considered when computing loss
            min_flux_for_metrics: filter sources by flux during test
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
            use_double_detect: whether to make up to two detections per tile rather than one
            use_checkerboard: whether to use dependent tiling
            reference_band: band to use for filtering sources
        """
        super().__init__()

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.image_normalizer = image_normalizer
        self.var_dist = var_dist
        self.mode_metrics = mode_metrics
        self.sample_metrics = sample_metrics
        self.sample_image_renders = sample_image_renders
        self.matcher = matcher
        self.min_flux_for_loss = min_flux_for_loss
        self.min_flux_for_metrics = min_flux_for_metrics
        assert self.min_flux_for_loss <= self.min_flux_for_metrics, "invalid threshold"
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.use_double_detect = use_double_detect
        self.use_checkerboard = use_checkerboard
        self.reference_band = reference_band

        # Generate all binary combinations for n^2 elements
        n = 2
        binary_combinations = list(itertools.product([0, 1], repeat=n * n))
        mask_patterns = torch.tensor(binary_combinations).view(-1, n, n)  # noqa: WPS114
        self.register_buffer("mask_patterns", mask_patterns)

        self.initialize_networks()

    def initialize_networks(self):
        assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"

        num_features = 256

        self.features_net = FeaturesNet(
            n_bands=len(self.image_normalizer.bands),
            ch_per_band=self.image_normalizer.num_channels_per_band(),
            num_features=num_features,
            double_downsample=(self.tile_slen == 4),
        )
        self.catalog_net = CatalogNet(
            num_features=num_features,
            out_channels=self.var_dist.n_params_per_source,
        )

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
            log_fluxes = (history_cat.on_fluxes.squeeze(3).sum(-1) + 1).log()
            history_encoding_lst = [
                history_cat["n_sources"].float(),  # detection history
                log_fluxes * history_cat["n_sources"],  # flux history
                centered_locs[..., 0] * history_cat["n_sources"],  # x history
                centered_locs[..., 1] * history_cat["n_sources"],  # y history
            ]
            masked_history_lst = [v * history_mask for v in history_encoding_lst]
            masked_history = torch.stack(masked_history_lst, dim=1)

        return torch.concat([detection_id, history_mask.unsqueeze(1), masked_history], dim=1)

    def sample(self, batch, use_mode=True):
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]
        ht, wt = h // self.tile_slen, w // self.tile_slen

        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)

        est_cat = None
        patterns_to_use = (0, 8, 12, 14) if self.use_checkerboard else (0,)

        for mask_pattern in self.mask_patterns[patterns_to_use, ...]:
            mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
            context1 = self.make_context(est_cat, mask)
            x_cat1 = self.catalog_net(x_features, context1)
            new_est_cat = self.var_dist.sample(x_cat1, use_mode=use_mode)
            new_est_cat["n_sources"] *= 1 - mask
            if est_cat is None:
                est_cat = new_est_cat
            else:
                est_cat["n_sources"] *= mask
                est_cat = est_cat.union(new_est_cat, disjoint=True)

        if self.use_double_detect:
            no_mask = torch.ones_like(mask)
            context2 = self.make_context(est_cat, no_mask, detection2=True)
            x_cat2 = self.catalog_net(x_features, context2)
            est_cat2 = self.var_dist.sample(x_cat2, use_mode=use_mode)
            # our loss function implies that the second detection is ignored for a tile
            # if the first detection is empty for that tile
            est_cat2["n_sources"] *= est_cat["n_sources"]
            est_cat = est_cat.union(est_cat2, disjoint=False)

        return est_cat

    def _compute_loss(self, batch, logging_name):
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]
        ht, wt = h // self.tile_slen, w // self.tile_slen

        # filter out undetectable sources and split catalog by flux
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        # TODO: move tile_cat filtering to the dataloader and from the encoder remove
        # `reference_band`, `min_flux_for_loss`, and `min_flux_for_metrics`
        # (metrics can be computed with the original full catalog if necessary)
        target_cat = target_cat.filter_by_flux(
            min_flux=self.min_flux_for_loss,
            band=self.reference_band,
        )
        # TODO: don't order the light sources by brightness; softmax instead
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat2 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=1
        )

        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)

        loss = torch.zeros_like(x_features[:, 0, :, :])

        # could use all the mask patterns but memory is tight
        patterns_to_use = torch.randperm(15)[:4] if self.use_checkerboard else (0,)

        for mask_pattern in self.mask_patterns[patterns_to_use, ...]:
            mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
            context1 = self.make_context(target_cat1, mask)
            x_cat1 = self.catalog_net(x_features, context1)
            loss11 = self.var_dist.compute_nll(x_cat1, target_cat1)

            # could upweight some patterns that are under-represented or limit loss to
            # the quarter of tiles that are actually used for sampling
            loss += loss11 * (1 - mask)

            if not self.use_checkerboard:
                break

        if self.use_double_detect:
            no_mask = torch.ones_like(mask)
            context2 = self.make_context(target_cat1, no_mask, detection2=True)
            x_cat2 = self.catalog_net(x_features, context2)
            loss22 = self.var_dist.compute_nll(x_cat2, target_cat2)
            loss22 *= target_cat1["n_sources"]
            loss += loss22

        # could normalize by the number of tile predictions, rather than number of tiles
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

    def sample_for_metrics(self, batch):
        target_tile_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])
        target_tile_cat = target_tile_cat.filter_by_flux(
            min_flux=self.min_flux_for_metrics,
            band=self.reference_band,
        )

        mode_tile_cat = self.sample(batch, use_mode=True).filter_by_flux(
            min_flux=self.min_flux_for_metrics,
            band=self.reference_band,
        )

        sample_tile_cat = None
        if self.sample_metrics is not None:
            sample_tile_cat = self.sample(batch, use_mode=False).filter_by_flux(
                min_flux=self.min_flux_for_metrics,
                band=self.reference_band,
            )

        return target_tile_cat, mode_tile_cat, sample_tile_cat

    def update_metrics(self, batch, batch_idx):
        target_tile_cat, mode_tile_cat, sample_tile_cat = self.sample_for_metrics(batch)

        target_cat = target_tile_cat.to_full_catalog()

        mode_cat = mode_tile_cat.to_full_catalog()
        mode_matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.mode_metrics.update(target_cat, mode_cat, mode_matching)

        if sample_tile_cat is not None:
            sample_cat = sample_tile_cat.to_full_catalog()
            sample_matching = self.matcher.match_catalogs(target_cat, sample_cat)
            self.sample_metrics.update(target_cat, sample_cat, sample_matching)

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
        if self.sample_metrics is not None:
            self.report_metrics(self.sample_metrics, "val/sample", show_epoch=True)
        self.report_metrics(self.sample_image_renders, "val/image_renders", show_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "test")
        self.update_metrics(batch, batch_idx)

    def on_test_epoch_end(self):
        self.report_metrics(self.mode_metrics, "test/mode", show_epoch=False)
        if self.sample_metrics is not None:
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
