import math
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch.distributions import Distribution
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from bliss.catalog import SourceType, TileCatalog
from bliss.convnet import ConditionalNet, MarginalNet
from bliss.data_augmentation import augment_data
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
        image_slen_in_tiles: int,
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
            image_slen_in_tiles: image height and width in tiles
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
        self.tiles_to_crop = tiles_to_crop
        self.image_normalizer = image_normalizer
        self.min_flux_threshold = min_flux_threshold
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.do_data_augmentation = do_data_augmentation

        self.tile_slen = tile_slen

        ch_per_band = self.image_normalizer.num_channels_per_band()
        n_params_per_source = sum(param.dim for param in self.dist_param_groups.values())
        self.marginal_net = MarginalNet(len(bands), ch_per_band, n_params_per_source)
        self.conditional_net = ConditionalNet(n_params_per_source)

        if compile_model:
            self.preprocess3d = torch.compile(self.preprocess3d)
            self.marginal_net = torch.compile(self.marginal_net)
            self.conditional_net = torch.compile(self.conditional_net)

        # metrics
        self.metrics = BlissMetrics(
            mode=MetricsMode.TILE, slack=slack, survey_bands=self.survey_bands
        )

        # make/store a checkerboard of tiles
        # https://stackoverflow.com/questions/72874737/how-to-make-a-checkerboard-in-pytorch
        arange = torch.arange(image_slen_in_tiles, device=self.device)
        mg = torch.meshgrid(arange, arange, indexing="ij")
        indices = torch.stack(mg)
        tile_cb = indices.sum(axis=0) % 2
        tile_cb = rearrange(tile_cb, "ht wt -> 1 1 ht wt")
        self.register_buffer("tile_cb", tile_cb)

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

    def encode_batch(self, batch, target_cat=None, vmode=False):
        x = self.image_normalizer.get_input_tensor(batch)
        x_cat_marginal, x_features = self.marginal_net(x)
        x_features = x_features.detach()  # helps with training stability

        if target_cat:  # training and validation loss
            detections = target_cat.n_sources > 0
        else:  # prediction and validation metrics
            on_dist = self.dist_param_groups["on_prob"].get_dist(x_cat_marginal[:, :, :, 0:1])
            if vmode:  # mode (of the variational distribution)
                detections = on_dist.mode
            else:  # sample
                detections = on_dist.sample()

        detections = detections.float().unsqueeze(1)

        detections1 = detections * self.tile_cb
        mask1 = self.tile_cb.expand([x_features.size(0), -1, -1, -1])
        x_cat1 = self.conditional_net(x_features, detections1, mask1)

        detections2 = detections * (1 - self.tile_cb)
        x_cat2 = self.conditional_net(x_features, detections2, 1 - mask1)

        tile_cb_view = rearrange(self.tile_cb, "1 1 ht wt -> 1 ht wt 1")
        x_cat_conditional = x_cat1 * (1 - tile_cb_view) + x_cat2 * tile_cb_view

        return self.get_predicted_dist(x_cat_marginal), self.get_predicted_dist(x_cat_conditional)

    def variational_mode(self, batch, use_marginal=True) -> TileCatalog:
        """Compute the mode of the variational distribution.

        Args:
            batch: model predictions
            use_marginal: whether to use the marginal or conditional predictions

        Returns:
            TileCatalog: Catalog based on predictions.
        """
        pred_marginal, pred_conditional = self.encode_batch(batch, vmode=True)
        # TODO: combine marginal and conditional, don't pick one or the other
        pred = pred_marginal if use_marginal else pred_conditional

        tile_is_on_array = pred["on_prob"].mode

        # populate est_catalog_dict with per-band (log) star fluxes
        star_fluxes = torch.stack(
            [pred[name].mode * tile_is_on_array for name in self.STAR_FLUX_NAMES], dim=3
        )

        # populate est_catalog_dict with source type
        galaxy_bools = pred["galaxy_prob"].mode
        star_bools = 1 - galaxy_bools
        source_type = SourceType.STAR * star_bools + SourceType.GALAXY * galaxy_bools

        # populate est_catalog_dict with galaxy parameters
        galsim_dists = [pred[f"galsim_{name}"] for name in self.GALSIM_NAMES]
        # for params with transformed distribution mode and median aren't implemented.
        # instead, we compute median using inverse cdf 0.5
        galsim_param_lst = [d.icdf(torch.tensor(0.5)) for d in galsim_dists]
        galaxy_params = torch.stack(galsim_param_lst, dim=3)

        # populate est_catalog_dict with per-band galaxy fluxes
        galaxy_fluxes = torch.stack(
            [pred[name].icdf(torch.tensor(0.5)) * tile_is_on_array for name in self.GAL_FLUX_NAMES],
            dim=3,
        )

        # we have to unsqueeze some tensors below because a TileCatalog can store multiple
        # light sources per tile, but we predict only one source per tile
        est_catalog_dict = {
            "locs": rearrange(pred["loc"].mode, "b ht wt d -> b ht wt 1 d"),
            "star_fluxes": rearrange(star_fluxes, "b ht wt d -> b ht wt 1 d"),
            "n_sources": tile_is_on_array,
            "source_type": rearrange(source_type, "b ht wt -> b ht wt 1 1"),
            "galaxy_params": rearrange(galaxy_params, "b ht wt d -> b ht wt 1 d"),
            "galaxy_fluxes": rearrange(galaxy_fluxes, "b ht wt d -> b ht wt 1 d"),
        }

        return TileCatalog(self.tile_slen, est_catalog_dict)

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]

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
        if target_cat.max_sources > 1:
            target_cat = target_cat.get_brightest_source_per_tile(band=2)
        if self.min_flux_threshold > 0:
            target_cat = target_cat.filter_tile_catalog_by_flux(min_flux=self.min_flux_threshold)

        pred_marginal, pred_conditional = self.encode_batch(batch, target_cat=target_cat)

        target_cat = target_cat.symmetric_crop(self.tiles_to_crop)
        marginal_loss_dict = self._get_loss(pred_marginal, target_cat)
        conditional_loss_dict = self._get_loss(pred_conditional, target_cat)

        # log all losses
        for k, v in marginal_loss_dict.items():
            self.log("{}-marginal/{}".format(logging_name, k), v, batch_size=batch_size)

        for k, v in conditional_loss_dict.items():
            self.log("{}-conditional/{}".format(logging_name, k), v, batch_size=batch_size)

        if logging_name == "val":
            self.log("val/loss", marginal_loss_dict["loss"] + conditional_loss_dict["loss"])

        if log_metrics or plot_images:
            est_tile_cat_marginal = self.variational_mode(batch, use_marginal=True)
            est_tile_cat_conditional = self.variational_mode(batch, use_marginal=False)

        # log all metrics
        if log_metrics:
            metrics = self.metrics(target_cat, est_tile_cat_marginal)
            for k, v in metrics.items():
                self.log("{}-marginal/{}".format(logging_name, k), v, batch_size=batch_size)
            metrics = self.metrics(target_cat, est_tile_cat_conditional)
            for k, v in metrics.items():
                self.log("{}-conditional/{}".format(logging_name, k), v, batch_size=batch_size)

        # log a grid of figures to the tensorboard
        if plot_images:
            batch_size = len(batch["images"])
            n_samples = min(int(math.sqrt(batch_size)) ** 2, 16)
            nrows = int(n_samples**0.5)  # for figure

            target_full_cat = target_cat.to_full_params()
            est_full_cat = est_tile_cat_marginal.to_full_params()
            wrong_idx = (est_full_cat.n_sources != target_full_cat.n_sources).nonzero()
            wrong_idx = wrong_idx.view(-1)[:n_samples]

            margin_px = self.tiles_to_crop * self.tile_slen
            fig = plot_detections(
                torch.squeeze(batch["images"], 2),
                target_full_cat,
                est_full_cat,
                nrows,
                wrong_idx,
                margin_px,
            )
            title_root = f"Epoch:{self.current_epoch}/"
            title = f"{title_root}{logging_name} images"
            if self.logger:
                self.logger.experiment.add_figure(title, fig)
            plt.close(fig)

        return marginal_loss_dict["loss"] + conditional_loss_dict["loss"]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        if self.do_data_augmentation:
            imgs = batch["images"][:, self.bands].unsqueeze(2)  # add extra dim for 5d input
            bgs = batch["background"][:, self.bands].unsqueeze(2)
            aug_input_images = [imgs, bgs]
            if self.image_normalizer.use_deconv_channel:
                assert (
                    "deconvolution" in batch
                ), "use_deconv_channel specified but deconvolution not present in data"
                aug_input_images.append(batch["background"][:, self.bands].unsqueeze(2))
            aug_input_image = torch.cat(aug_input_images, dim=2)

            aug_output_image, tile = augment_data(batch["tile_catalog"], aug_input_image)
            batch["images"] = aug_output_image[:, :, 0, :, :]
            batch["background"] = aug_output_image[:, :, 1, :, :]
            batch["tile_catalog"] = tile
            if self.image_normalizer.use_deconv_channel:
                batch["deconvolution"] = aug_output_image[:, :, 2, :, :]

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
        with torch.no_grad():
            pred_marginal, _ = self.encode_batch(batch)
            est_cat = self.variational_mode(batch, use_marginal=True)
        return {
            "est_cat": est_cat,
            "images": batch["images"],
            "background": batch["background"],
            "pred": pred_marginal,
        }
