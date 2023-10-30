from copy import deepcopy
from typing import Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from bliss.catalog import TileCatalog
from bliss.convnet import CatalogNet, FeaturesNet
from bliss.data_augmentation import augment_batch
from bliss.image_normalizer import ImageNormalizer
from bliss.metrics import BlissMetrics, MetricsMode
from bliss.plotting import plot_detections
from bliss.unconstrained_dists import (
    UnconstrainedBernoulli,
    UnconstrainedLogitNormal,
    UnconstrainedLogNormal,
    UnconstrainedTDBN,
)
from bliss.variational_layer import VariationalLayer


class Encoder(pl.LightningModule):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
        bands: list,
        survey_bands: list,
        tile_slen: int,
        tiles_to_crop: int,
        image_normalizer: ImageNormalizer,
        slack: float = 1.0,
        min_flux_threshold: float = 0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        do_data_augmentation: bool = False,
        compile_model: bool = False,
    ):
        """Initializes DetectionEncoder.

        Args:
            bands: specified band-pass filters
            survey_bands: all band-pass filters available for this survey
            tile_slen: dimension in pixels of a square tile
            tiles_to_crop: margin of tiles not to use for computing loss
            image_normalizer: object that applies input transforms to images
            slack: Slack to use when matching locations for validation metrics.
            min_flux_threshold: Sources with a lower flux will not be considered when computing loss
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
            do_data_augmentation: used for determining whether or not do data augmentation
            compile_model: compile model for potential performance improvements
        """
        super().__init__()
        self.save_hyperparameters(ignore=["image_normalizer"])

        self.bands = bands
        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.tiles_to_crop = tiles_to_crop
        self.image_normalizer = image_normalizer
        self.min_flux_threshold = min_flux_threshold
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.do_data_augmentation = do_data_augmentation

        ch_per_band = self.image_normalizer.num_channels_per_band()
        n_params_per_source = sum(param.dim for param in self.dist_param_groups.values())
        assert tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        self.features_net = FeaturesNet(
            len(bands),
            ch_per_band,
            n_params_per_source,
            double_downsample=(tile_slen == 4),
        )
        self.catalog_net = CatalogNet(n_params_per_source)

        if compile_model:
            self.features_net = torch.compile(self.features_net)
            self.catalog_net = torch.compile(self.catalog_net)

        # metrics
        self.metrics = BlissMetrics(
            mode=MetricsMode.TILE, slack=slack, survey_bands=self.survey_bands
        )

        self.automatic_optimization = False

    @property
    def dist_param_groups(self):
        d = {
            "on_prob": UnconstrainedBernoulli(),
            "loc": UnconstrainedTDBN(),
            "galaxy_prob": UnconstrainedBernoulli(),
            # galsim parameters
            "galsim_disk_frac": UnconstrainedLogitNormal(),
            "galsim_beta_radians": UnconstrainedLogitNormal(high=torch.pi),
            "galsim_disk_q": UnconstrainedLogitNormal(),
            "galsim_a_d": UnconstrainedLogNormal(),
            "galsim_bulge_q": UnconstrainedLogitNormal(),
            "galsim_a_b": UnconstrainedLogNormal(),
        }
        for band in self.survey_bands:
            d[f"star_flux_{band}"] = UnconstrainedLogNormal()
        for band in self.survey_bands:
            d[f"galaxy_flux_{band}"] = UnconstrainedLogNormal()
        return d

    def _get_checkboard(self, ht, wt):
        # make/store a checkerboard of tiles
        # https://stackoverflow.com/questions/72874737/how-to-make-a-checkerboard-in-pytorch
        arange_ht = torch.arange(ht, device=self.device)
        arange_wt = torch.arange(wt, device=self.device)
        mg = torch.meshgrid(arange_ht, arange_wt, indexing="ij")
        indices = torch.stack(mg)
        tile_cb = indices.sum(axis=0) % 2
        return rearrange(tile_cb, "ht wt -> 1 1 ht wt")

    def sample(self, batch, cat_type="joint1", use_mode=False, layer2=False) -> TileCatalog:
        """Sample the variational distribution.

        Args:
            batch: (transformed) input data
            cat_type: whether to use the marginal or joint predictions
            use_mode: whether to use the mode of the distribution instead of random sampling
            layer2: whether to sample the second layer of light sources

        Returns:
            TileCatalog: Sampled catalog
        """
        assert cat_type in {"marginal", "conditional", "joint1"}, "joint2 not supported"

        def detection_callback(x_cat_marginal):  # noqa: WPS430
            on_dist = self.dist_param_groups["on_prob"].get_dist(x_cat_marginal[:, :, :, 0:1])
            return on_dist.mode if use_mode else on_dist.sample()

        preds = self.infer(batch, detection_callback, layer2=layer2)

        pred = preds[cat_type]
        est_cat = pred.sample(use_mode=use_mode)

        if cat_type == "joint1":
            ttc = self.tiles_to_crop
            if ttc > 0:
                tile_cb = preds["tile_cb"].squeeze(1)
                tile_cb = tile_cb[:, ttc:-ttc, ttc:-ttc]
                md = preds["marginal_detections"][:, ttc:-ttc, ttc:-ttc]
            est_cat.n_sources *= 1 - tile_cb
            est_cat.n_sources += tile_cb * md

        return est_cat

    def make_layer(self, x_cat):
        ttc = self.tiles_to_crop
        if ttc > 0:
            x_cat = x_cat[:, ttc:-ttc, ttc:-ttc, :]

        split_sizes = [v.dim for v in self.dist_param_groups.values()]
        dist_params_split = torch.split(x_cat, split_sizes, 3)
        names = self.dist_param_groups.keys()
        pred = dict(zip(names, dist_params_split))

        for k, v in pred.items():
            pred[k] = self.dist_param_groups[k].get_dist(v)

        return VariationalLayer(pred, self.survey_bands, self.tile_slen)

    def get_features(self, batch):
        x = self.image_normalizer.get_input_tensor(batch)
        return self.features_net(x)

    def get_context(self, iter_name, tile_cat):
        assert iter_name in {"marginal", "white", "black", "second"}
        bs, ht, wt = tile_cat.n_sources.shape

        pass_context = torch.zeros(size=(bs, 1, ht, wt), device=self.device)
        if iter_name == "second":
            pass_context.fill_(1.0)

        tile_cb = self._get_checkboard(ht, wt)
        white_mask = tile_cb.expand([bs, -1, -1, -1])
        color_mask = 1 - white_mask if iter_name == "black" else white_mask
        if iter_name in {"marginal", "second"}:
            color_mask.fill_(0.0)

        # we may want to use richer conditioning information in the future;
        # e.g., a residual image based on the catalog so far
        detection_history = (tile_cat.n_sources > 0).unsqueeze(1)
        log_flux_history = (tile_cat.get_fluxes_of_on_sources().sum(-1) + 1).log()
        log_flux_history = rearrange(log_flux_history, "b ht wt 1 -> b 1 ht wt")

        return torch.cat([color_mask, pass_context, detection_history, log_flux_history], dim=1)

    def interleave_catalogs(self, x_cat1, x_cat2):
        _, ht, wt, _ = x_cat1.shape
        tile_cb = self._get_checkboard(ht, wt)
        tile_cb_view = rearrange(tile_cb, "1 1 ht wt -> 1 ht wt 1")
        return x_cat1 * (1 - tile_cb_view) + x_cat2 * tile_cb_view

    def _generic_step(self, batch, logging_name, log_metrics=False, plot_images=False):
        batch_size = batch["images"].size(0)
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        # Filter by detectable sources and brightest source per tile
        if self.min_flux_threshold > 0:
            target_cat = target_cat.filter_tile_catalog_by_flux(min_flux=self.min_flux_threshold)

        if self.training:
            opt = self.optimizers()
            opt.zero_grad()

        x_features = self.get_features(batch)

        # predict first layer (brightest) of light sources; marginal
        target_cat1 = target_cat.get_brightest_sources_per_tile(band=2, exclude_num=0)
        empty_cat = deepcopy(target_cat1)
        empty_cat.n_sources.fill_(0)
        context_marginal = self.get_context("marginal", empty_cat)
        x_cat_marginal = self.catalog_net(x_features, context_marginal)
        pred_marginal = self.make_layer(x_cat_marginal)
        target_cat1_cropped = target_cat1.symmetric_crop(self.tiles_to_crop)
        marginal_loss_dict = pred_marginal.compute_nll(target_cat1_cropped)

        if self.training:
            opt = self.optimizers()
            self.manual_backward(marginal_loss_dict["loss"] / 2)
            opt.step()
            opt.zero_grad()

        # predict first layer (brightest) of light sources; conditional
        x_features = self.get_features(batch)

        target_cat_white = deepcopy(target_cat1)
        tile_cb = self._get_checkboard(target_cat.n_tiles_h, target_cat.n_tiles_w).squeeze(1)
        target_cat_white.n_sources *= tile_cb
        context_white = self.get_context("white", target_cat_white)
        x_cat_white = self.catalog_net(x_features, context_white)

        target_cat_black = deepcopy(target_cat1)
        target_cat_black.n_sources *= 1 - tile_cb
        context_black = self.get_context("black", target_cat_black)
        x_cat_black = self.catalog_net(x_features, context_black)

        x_cat_cond = self.interleave_catalogs(x_cat_white, x_cat_black)
        pred_cond = self.make_layer(x_cat_cond)
        conditional_loss_dict = pred_cond.compute_nll(target_cat1_cropped)

        if self.training:
            opt = self.optimizers()
            self.manual_backward(conditional_loss_dict["loss"] / 2)
            opt.step()
            opt.zero_grad()

        # predict second layer (next brightest) of light sources; marginal
        x_features = self.get_features(batch)

        target_cat2 = target_cat.get_brightest_sources_per_tile(band=2, exclude_num=1)
        context2 = self.get_context("second", target_cat1)
        x_cat_second = self.catalog_net(x_features, context2)
        pred_second = self.make_layer(x_cat_second)
        target_cat2_cropped = target_cat2.symmetric_crop(self.tiles_to_crop)
        second_loss_dict = pred_second.compute_nll(target_cat2_cropped)

        if self.training:
            opt = self.optimizers()
            self.manual_backward(second_loss_dict["loss"])
            opt.step()

        # log layer1 losses
        for k, v in marginal_loss_dict.items():
            loss = (v + conditional_loss_dict[k]) / 2  # we make two predictions for each tile
            self.log("{}-layer1/{}".format(logging_name, k), loss, batch_size=batch_size)
            self.log("{}-layer1-marginal/{}".format(logging_name, k), v, batch_size=batch_size)

        # log layer2 losses
        for k, v in second_loss_dict.items():
            self.log("{}-layer2/{}".format(logging_name, k), v, batch_size=batch_size)

        # log sum of losses
        for k, v in marginal_loss_dict.items():
            loss = (v + conditional_loss_dict[k]) / 2
            loss += second_loss_dict[k]
            self.log("{}/{}".format(logging_name, k), loss, batch_size=batch_size)

        # log metrics
        if log_metrics:
            est_tile_cat = pred_marginal.sample(use_mode=True)
            metrics = self.metrics(target_cat1_cropped, est_tile_cat)
            for k, v in metrics.items():
                metric_name = "{}-metrics-marginal/{}".format(logging_name, k)
                self.log(metric_name, v, batch_size=batch_size)

            est_tile_cat = pred_cond.sample(use_mode=True)
            metrics = self.metrics(target_cat1_cropped, est_tile_cat)
            for k, v in metrics.items():
                metric_name = "{}-metrics-cond/{}".format(logging_name, k)
                self.log(metric_name, v, batch_size=batch_size)

            est_tile_cat = pred_second.sample(use_mode=True)
            metrics = self.metrics(target_cat1_cropped, est_tile_cat)
            for k, v in metrics.items():
                metric_name = "{}-metrics-second/{}".format(logging_name, k)
                self.log(metric_name, v, batch_size=batch_size)

        # log a grid of figures to the tensorboard
        if plot_images:
            est_tile_cat = pred_marginal.sample(use_mode=True)
            mp = self.tiles_to_crop * self.tile_slen
            fig = plot_detections(batch["images"], target_cat1_cropped, est_tile_cat, margin_px=mp)
            title = f"Epoch:{self.current_epoch}/{logging_name} images"
            if self.logger:
                self.logger.experiment.add_figure(title, fig)
            plt.close(fig)

        first_pass_loss = (marginal_loss_dict["loss"] + conditional_loss_dict["loss"]) / 2
        return first_pass_loss + second_loss_dict["loss"]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        if self.do_data_augmentation:
            augment_batch(batch)

        return self._generic_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        # only plot images on the first batch of every 10th epoch
        epoch = self.trainer.current_epoch
        plot_images = batch_idx == 0 and (epoch % 10 == 0 or epoch == self.trainer.max_epochs - 1)
        self._generic_step(batch, "val", log_metrics=True, plot_images=plot_images)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._generic_step(batch, "test", log_metrics=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Pytorch lightning method."""

        def marginal_detections(x_cat_marginal):  # noqa: WPS430
            return self.dist_param_groups["on_prob"].get_dist(x_cat_marginal[:, :, :, 0:1]).mode

        preds = self.infer(batch, marginal_detections, layer2=False)
        return {
            "est_cat": self.sample(batch, use_mode=True),
            # a marginal catalog isn't really what we want here, perhaps
            # we should return samples from the variation distribution instead
            "pred": preds["marginal"],
        }

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]
