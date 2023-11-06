from copy import deepcopy
from typing import Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from bliss.catalog import TileCatalog
from bliss.convnet import CatalogNet, ContextNet, FeaturesNet
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
        two_layers: bool = False,
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
            two_layers: whether to make up to two detections per tile rather than one
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
        self.two_layers = two_layers

        ch_per_band = self.image_normalizer.num_channels_per_band()
        n_params_per_source = sum(param.dim for param in self.dist_param_groups.values())
        assert tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        num_features = 256
        self.features_net = FeaturesNet(
            len(bands),
            ch_per_band,
            num_features,
            double_downsample=(tile_slen == 4),
        )
        self.marginal_net = CatalogNet(num_features, n_params_per_source)
        self.checkerboard_net = ContextNet(num_features, n_params_per_source)
        if self.two_layers:
            self.second_net = CatalogNet(num_features, n_params_per_source)

        if compile_model:
            self.features_net = torch.compile(self.features_net)
            self.marginal_net = torch.compile(self.marginal_net)
            self.checkerboard_net = torch.compile(self.checkerboard_net)
            if self.two_layers:
                self.second_net = torch.compile(self.second_net)

        # metrics
        self.metrics = BlissMetrics(
            mode=MetricsMode.TILE, slack=slack, survey_bands=self.survey_bands
        )

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

    def make_layer(self, x_cat):
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

    def infer_conditional(self, x_features, history_cat, history_mask):
        masked_cat = deepcopy(history_cat)
        # masks not just n_sources; n_sources controls access to all fields
        masked_cat.n_sources *= history_mask

        # we may want to use richer conditioning information in the future;
        # e.g., a residual image based on the catalog so far
        detection_history = masked_cat.n_sources > 0

        context = torch.stack([detection_history, history_mask], dim=1).float()
        x_cat = self.checkerboard_net(x_features, context)
        return self.make_layer(x_cat)

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
        tile_cb = self._get_checkboard(ht, wt).squeeze(1)
        pred = {}

        x_features = self.get_features(batch)

        x_cat_marginal = self.marginal_net(x_features)
        x_features = x_features.detach()  # is this helpful? doing it here to match old code
        pred["marginal"] = self.make_layer(x_cat_marginal)

        history_cat = history_callback(pred["marginal"])

        white_history_mask = tile_cb.expand([batch_size, -1, -1])
        pred["white"] = self.infer_conditional(x_features, history_cat, white_history_mask)
        pred["black"] = self.infer_conditional(x_features, history_cat, 1 - white_history_mask)

        if self.two_layers:
            x_cat_second = self.second_net(x_features)
            pred["second"] = self.make_layer(x_cat_second)

        pred["history_cat"] = history_cat
        pred["x_features"] = x_features
        pred["white_history_mask"] = white_history_mask

        return pred

    def log_metrics(self, target_cats, pred, logging_name, images, plot_images):
        batch_size = target_cats["layer1"].n_sources.size(0)
        target_cat1_cropped = target_cats["layer1"].symmetric_crop(self.tiles_to_crop)

        # marginal metrics, layer 1
        est_cat_m = pred["marginal"].sample(use_mode=True)
        ecm_cropped = est_cat_m.symmetric_crop(self.tiles_to_crop)
        metrics_marginal = self.metrics(target_cat1_cropped, ecm_cropped)
        for k in ("f1", "detection_precision", "detection_recall"):
            metric_name = "{}-layer1/{}-marginal".format(logging_name, k)
            self.log(metric_name, metrics_marginal[k], batch_size=batch_size)

        # joint metrics, layer 1
        pred_white_notruth = self.infer_conditional(
            pred["x_features"], est_cat_m, pred["white_history_mask"]
        )
        est_cat_white = pred_white_notruth.sample(use_mode=True)
        est_cat_joint = self.interleave_catalogs(
            est_cat_m, est_cat_white, pred["white_history_mask"]
        )
        ecj_cropped = est_cat_joint.symmetric_crop(self.tiles_to_crop)
        metrics_joint = self.metrics(target_cat1_cropped, ecj_cropped)
        for k in ("f1", "detection_precision", "detection_recall"):
            metric_name = "{}-layer1/{}-joint".format(logging_name, k)
            self.log(metric_name, metrics_joint[k], batch_size=batch_size)

        if self.two_layers:
            target_cat2_cropped = target_cats["layer2"].symmetric_crop(self.tiles_to_crop)
            est_cat_s = pred["second"].sample(use_mode=True).symmetric_crop(self.tiles_to_crop)
            metrics_second = self.metrics(target_cat2_cropped, est_cat_s)
            for k in ("f1", "detection_precision", "detection_recall"):
                metric_name = "{}-layer2/{}".format(logging_name, k)
                self.log(metric_name, metrics_second[k], batch_size=batch_size)

        # log a grid of figures to the tensorboard
        if plot_images:
            mp = self.tiles_to_crop * self.tile_slen
            fig = plot_detections(images, target_cat1_cropped, ecj_cropped, margin_px=mp)
            title = f"Epoch:{self.current_epoch}/{logging_name} images"
            if self.logger:
                self.logger.experiment.add_figure(title, fig)
            plt.close(fig)

    def _generic_step(self, batch, logging_name, log_metrics=False, plot_images=False):
        batch_size = batch["images"].size(0)
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        # Filter by detectable sources and brightest source per tile
        if self.min_flux_threshold > 0:
            target_cat = target_cat.filter_tile_catalog_by_flux(min_flux=self.min_flux_threshold)

        target_cat1 = target_cat.get_brightest_sources_per_tile(band=2, exclude_num=0)
        truth_callback = lambda _: target_cat1  # noqa: E731
        pred = self.infer(batch, truth_callback)

        border_mask = torch.ones_like(target_cat.n_sources)
        ttc = self.tiles_to_crop
        border_mask = pad(pad(border_mask, [-ttc, -ttc, -ttc, -ttc]), [ttc, ttc, ttc, ttc])
        white_loss_mask = border_mask * (1 - pred["white_history_mask"])
        black_loss_mask = border_mask * pred["white_history_mask"]

        marginal_loss_dict = pred["marginal"].compute_nll(target_cat1, border_mask)
        white_loss_dict = pred["white"].compute_nll(target_cat1, white_loss_mask)
        black_loss_dict = pred["black"].compute_nll(target_cat1, black_loss_mask)

        if self.two_layers:
            target_cat2 = target_cat.get_brightest_sources_per_tile(band=2, exclude_num=1)
            second_loss_dict = pred["second"].compute_nll(target_cat2, border_mask)

        # log layer1 losses, divide by 2 because we make two predictions for each tile
        for k in ("loss", "counter_loss"):
            loss_l1 = (marginal_loss_dict[k] + white_loss_dict[k] + black_loss_dict[k]) / 2
            loss_l1 /= border_mask.sum()  # per tile loss
            self.log(f"{logging_name}-layer1/_{k}", loss_l1, batch_size=batch_size)

        # log layer2 losses
        if self.two_layers:
            for k in ("loss", "counter_loss"):
                loss_l2 = second_loss_dict[k] / border_mask.sum()
                self.log(f"{logging_name}-layer2/_{k}", loss_l2, batch_size=batch_size)

        # log metrics
        assert log_metrics or not plot_images, "plot_images requires log_metrics"
        if log_metrics:
            target_cats = {"layer1": target_cat1}
            if self.two_layers:
                target_cats["layer2"] = target_cat2
            self.log_metrics(
                target_cats, pred, logging_name, batch["images"], plot_images=plot_images
            )

        loss = marginal_loss_dict["loss"]
        loss += white_loss_dict["loss"] + black_loss_dict["loss"]
        loss /= 2
        if self.two_layers:
            loss += second_loss_dict["loss"]
        loss /= border_mask.sum()
        return loss

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

        def marginal_detections(pred_marginal):  # noqa: WPS430
            return pred_marginal["on_prob"].mode

        pred = self.infer(batch, marginal_detections)
        white_cat = pred["white"].sample(use_mode=True)
        est_cat = self.interleave_catalogs(
            pred["history_cat"], white_cat, pred["white_history_mask"]
        )
        return {
            "est_cat": est_cat,
            # a marginal catalog isn't really what we want here, perhaps
            # we should return samples from the variation distribution instead
            "pred": None,
        }

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]
