from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch.distributions import Distribution
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from bliss.catalog import SourceType, TileCatalog
from bliss.convnet import DetectionNet, FeaturesNet
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


class Encoder(pl.LightningModule):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    GALSIM_NAMES = ["disk_frac", "beta_radians", "disk_q", "a_d", "bulge_q", "a_b"]

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

        self.STAR_FLUX_NAMES = [f"star_flux_{bnd}" for bnd in survey_bands]  # ordered by BANDS
        self.GAL_FLUX_NAMES = [f"galaxy_flux_{bnd}" for bnd in survey_bands]  # ordered by BANDS

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
        self.detection_net = DetectionNet(n_params_per_source)

        if compile_model:
            self.features_net = torch.compile(self.features_net)
            self.detection_net = torch.compile(self.detection_net)

        # metrics
        self.metrics = BlissMetrics(
            mode=MetricsMode.TILE, slack=slack, survey_bands=self.survey_bands
        )

    def _get_checkboard(self, ht, wt):
        # make/store a checkerboard of tiles
        # https://stackoverflow.com/questions/72874737/how-to-make-a-checkerboard-in-pytorch
        arange_ht = torch.arange(ht, device=self.device)
        arange_wt = torch.arange(wt, device=self.device)
        mg = torch.meshgrid(arange_ht, arange_wt, indexing="ij")
        indices = torch.stack(mg)
        tile_cb = indices.sum(axis=0) % 2
        return rearrange(tile_cb, "ht wt -> 1 1 ht wt")

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
        for star_flux in self.STAR_FLUX_NAMES:
            d[star_flux] = UnconstrainedLogNormal()
        for gal_flux in self.GAL_FLUX_NAMES:
            d[gal_flux] = UnconstrainedLogNormal()
        return d

    def get_predicted_dist(self, x):
        ttc = self.tiles_to_crop
        if ttc > 0:
            x = x[:, ttc:-ttc, ttc:-ttc, :]

        split_sizes = [v.dim for v in self.dist_param_groups.values()]
        dist_params_split = torch.split(x, split_sizes, 3)
        names = self.dist_param_groups.keys()
        pred = dict(zip(names, dist_params_split))

        for k, v in pred.items():
            pred[k] = self.dist_param_groups[k].get_dist(v)

        return pred

    def infer(self, batch, detection_callback, layer2=False):
        x = self.image_normalizer.get_input_tensor(batch)
        x_features = self.features_net(x)

        context = torch.zeros_like(x_features[:, 0:3, :, :])
        if layer2:
            context[:, 2] = 1.0

        x_cat_marginal = self.detection_net(x_features, context)

        marginal_detections = detection_callback(x_cat_marginal)
        detections = marginal_detections.float().unsqueeze(1)

        tile_cb = self._get_checkboard(detections.size(2), detections.size(3))
        detections1 = detections * tile_cb
        mask1 = tile_cb.expand([x_features.size(0), -1, -1, -1])
        pass_context = torch.ones_like(mask1) if layer2 else torch.zeros_like(mask1)
        context1 = torch.cat([detections1, mask1, pass_context], dim=1)
        x_cat1 = self.detection_net(x_features, context1)

        detections2 = detections * (1 - tile_cb)
        context2 = torch.cat([detections2, 1 - mask1, pass_context], dim=1)
        x_cat2 = self.detection_net(x_features, context2)

        tile_cb_view = rearrange(tile_cb, "1 1 ht wt -> 1 ht wt 1")
        x_cat_conditional = x_cat1 * (1 - tile_cb_view) + x_cat2 * tile_cb_view

        x_cat_joint1 = x_cat_marginal * tile_cb_view + x_cat1 * (1 - tile_cb_view)
        x_cat_joint2 = x_cat_marginal * (1 - tile_cb_view) + x_cat2 * tile_cb_view

        return {
            "marginal": self.get_predicted_dist(x_cat_marginal),
            "conditional": self.get_predicted_dist(x_cat_conditional),
            "joint1": self.get_predicted_dist(x_cat_joint1),
            "joint2": self.get_predicted_dist(x_cat_joint2),
            "tile_cb": tile_cb,
            "marginal_detections": marginal_detections,
        }

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

        locs = pred["loc"].mode if use_mode else pred["loc"].sample().squeeze(0)
        est_cat = {"locs": locs}

        # populate catalog with per-band (log) star fluxes
        sf_preds = [pred[name] for name in self.STAR_FLUX_NAMES]
        sf_lst = [p.mode if use_mode else p.sample() for p in sf_preds]
        est_cat["star_fluxes"] = torch.stack(sf_lst, dim=3)

        # populate catalog with source type
        galaxy_bools = pred["galaxy_prob"].mode if use_mode else pred["galaxy_prob"].sample()
        galaxy_bools = galaxy_bools.unsqueeze(3)
        star_bools = 1 - galaxy_bools
        est_cat["source_type"] = SourceType.STAR * star_bools + SourceType.GALAXY * galaxy_bools

        # populate catalog with galaxy parameters
        gs_dists = [pred[f"galsim_{name}"] for name in self.GALSIM_NAMES]
        gs_param_lst = [d.icdf(torch.tensor(0.5)) if use_mode else d.sample() for d in gs_dists]
        est_cat["galaxy_params"] = torch.stack(gs_param_lst, dim=3)

        # populate catalog with per-band galaxy fluxes
        gf_dists = [pred[name] for name in self.GAL_FLUX_NAMES]
        gf_lst = [d.icdf(torch.tensor(0.5)) if use_mode else d.sample() for d in gf_dists]
        est_cat["galaxy_fluxes"] = torch.stack(gf_lst, dim=3)

        # we have to unsqueeze these tensors because a TileCatalog can store multiple
        # light sources per tile, but we sample only one source per tile
        for k, v in est_cat.items():
            est_cat[k] = v.unsqueeze(3)

        # n_sources is not unsqueezed because it is a single integer per tile regardless of
        # how many light sources are stored per tile
        est_cat["n_sources"] = pred["on_prob"].mode if use_mode else pred["on_prob"].sample()
        tile_cb = preds["tile_cb"].squeeze(1)
        if cat_type == "joint1":
            ttc = self.tiles_to_crop
            if ttc > 0:
                tile_cb = tile_cb[:, ttc:-ttc, ttc:-ttc]
                md = preds["marginal_detections"][:, ttc:-ttc, ttc:-ttc]
            est_cat["n_sources"] *= 1 - tile_cb
            est_cat["n_sources"] += tile_cb * md

        return TileCatalog(self.tile_slen, est_cat)

    def _get_loss(self, pred: Dict[str, Distribution], true_tile_cat: TileCatalog):
        loss_with_components = {}

        # counter loss
        counter_loss = -pred["on_prob"].log_prob(true_tile_cat.n_sources)
        loss = counter_loss
        loss_with_components["counter_loss"] = counter_loss.mean()

        # all the squeezing/rearranging below is because a TileCatalog can store multiple
        # light sources per tile, which is annoying here, but helpful for storing samples
        # and real catalogs. Still, there may be a better way.

        # location loss
        true_locs = true_tile_cat.locs.squeeze(3)
        locs_loss = -pred["loc"].log_prob(true_locs)
        locs_loss *= true_tile_cat.n_sources
        loss += locs_loss
        loss_with_components["locs_loss"] = locs_loss.sum() / true_tile_cat.n_sources.sum()

        # star/galaxy classification loss
        true_gal_bools = rearrange(true_tile_cat.galaxy_bools, "b ht wt 1 1 -> b ht wt")
        binary_loss = -pred["galaxy_prob"].log_prob(true_gal_bools)
        binary_loss *= true_tile_cat.n_sources
        loss += binary_loss
        loss_with_components["binary_loss"] = binary_loss.sum() / true_tile_cat.n_sources.sum()

        # flux losses
        true_star_bools = rearrange(true_tile_cat.star_bools, "b ht wt 1 1 -> b ht wt")
        star_fluxes = rearrange(true_tile_cat["star_fluxes"], "b ht wt 1 bnd -> b ht wt bnd")
        galaxy_fluxes = rearrange(true_tile_cat["galaxy_fluxes"], "b ht wt 1 bnd -> b ht wt bnd")

        # only compute loss over bands we're using
        star_bands = [self.STAR_FLUX_NAMES[band] for band in self.bands]
        gal_bands = [self.GAL_FLUX_NAMES[band] for band in self.bands]
        for i, star_name, gal_name in zip(self.bands, star_bands, gal_bands):
            # star flux loss
            star_flux_loss = -pred[star_name].log_prob(star_fluxes[..., i] + 1e-9) * true_star_bools
            loss += star_flux_loss
            loss_with_components[star_name] = star_flux_loss.sum() / true_star_bools.sum()

            # galaxy flux loss
            gal_flux_loss = -pred[gal_name].log_prob(galaxy_fluxes[..., i] + 1e-9) * true_gal_bools
            loss += gal_flux_loss
            loss_with_components[gal_name] = gal_flux_loss.sum() / true_gal_bools.sum()

        # galaxy properties loss
        galsim_true_vals = rearrange(true_tile_cat["galaxy_params"], "b ht wt 1 d -> b ht wt d")
        for i, param_name in enumerate(self.GALSIM_NAMES):
            galsim_pn = f"galsim_{param_name}"
            loss_term = -pred[galsim_pn].log_prob(galsim_true_vals[..., i] + 1e-9) * true_gal_bools
            loss += loss_term
            loss_with_components[galsim_pn] = loss_term.sum() / true_gal_bools.sum()

        loss_with_components["loss"] = loss.mean()

        return loss_with_components

    def _generic_step(self, batch, logging_name, log_metrics=False, plot_images=False):
        batch_size = batch["images"].size(0)
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        # Filter by detectable sources and brightest source per tile
        if self.min_flux_threshold > 0:
            target_cat = target_cat.filter_tile_catalog_by_flux(min_flux=self.min_flux_threshold)

        # predict first layer (brightest) of light sources
        target_cat1 = target_cat.get_brightest_sources_per_tile(band=2, exclude_num=0)
        preds1 = self.infer(batch, lambda _: target_cat1.n_sources > 0)
        target_cat1 = target_cat1.symmetric_crop(self.tiles_to_crop)
        marginal_loss_dict1 = self._get_loss(preds1["marginal"], target_cat1)
        conditional_loss_dict1 = self._get_loss(preds1["conditional"], target_cat1)

        # predict second layer (next brightest) of light sources
        target_cat2 = target_cat.get_brightest_sources_per_tile(band=2, exclude_num=1)
        preds2 = self.infer(batch, lambda _: target_cat2.n_sources > 0, layer2=True)
        target_cat2 = target_cat2.symmetric_crop(self.tiles_to_crop)
        marginal_loss_dict2 = self._get_loss(preds2["marginal"], target_cat2)
        conditional_loss_dict2 = self._get_loss(preds2["conditional"], target_cat2)

        # log all losses
        for k, ml in marginal_loss_dict1.items():
            v = (ml + conditional_loss_dict1[k]) / 2  # we make two predictions for each tile
            self.log("{}-layer1/{}".format(logging_name, k), v, batch_size=batch_size)
            self.log("{}-layer1-marginal/{}".format(logging_name, k), ml, batch_size=batch_size)

        # log all losses
        for k, ml in marginal_loss_dict2.items():
            v = (ml + conditional_loss_dict2[k]) / 2  # we make two predictions for each tile
            self.log("{}-layer2/{}".format(logging_name, k), v, batch_size=batch_size)
            self.log("{}-layer2-marginal/{}".format(logging_name, k), ml, batch_size=batch_size)

        # log sum
        for k, ml in marginal_loss_dict1.items():
            v = ml + conditional_loss_dict1[k]
            v += marginal_loss_dict2[k] + conditional_loss_dict2[k]
            v /= 2
            self.log("{}/{}".format(logging_name, k), v, batch_size=batch_size)

        # log all metrics
        if log_metrics:
            est_tile_cat = self.sample(batch, use_mode=True, layer2=False)
            metrics = self.metrics(target_cat1, est_tile_cat)
            for k, v in metrics.items():
                metric_name = "{}-layer1-metrics/{}".format(logging_name, k)
                self.log(metric_name, v, batch_size=batch_size)

            est_tile_cat = self.sample(batch, use_mode=True, layer2=True)
            metrics = self.metrics(target_cat2, est_tile_cat)
            for k, v in metrics.items():
                metric_name = "{}-layer2-metrics/{}".format(logging_name, k)
                self.log(metric_name, v, batch_size=batch_size)

        # log a grid of figures to the tensorboard
        if plot_images:
            target_full_cat = target_cat1.to_full_params()

            est_tile_cat = self.sample(batch, cat_type="joint1", use_mode=True)
            est_full_cat = est_tile_cat.to_full_params()

            mp = self.tiles_to_crop * self.tile_slen
            fig = plot_detections(batch["images"], target_full_cat, est_full_cat, margin_px=mp)

            title = f"Epoch:{self.current_epoch}/{logging_name} images"
            if self.logger:
                self.logger.experiment.add_figure(title, fig)
            plt.close(fig)

        return (
            marginal_loss_dict1["loss"]
            + conditional_loss_dict1["loss"]
            + marginal_loss_dict2["loss"]
            + conditional_loss_dict2["loss"]
        ) / 2

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
