import math
import warnings
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from bliss.backbone import Backbone
from bliss.catalog import FullCatalog, SourceType, TileCatalog
from bliss.data_augmentation import augment_data
from bliss.metrics import BlissMetrics, MetricsMode
from bliss.plotting import plot_detections
from bliss.transforms import log_transform, rolling_z_score
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
        architecture: DictConfig,
        bands: list,
        survey_bands: list,
        tile_slen: int,
        tiles_to_crop: int,
        slack: float = 1.0,
        min_flux_threshold: float = 0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        input_transform_params: Optional[dict] = None,
        do_data_augmentation: bool = False,
        compile_model: bool = False,
    ):
        """Initializes DetectionEncoder.

        Args:
            architecture: yaml to specifying the encoder network architecture
            bands: specified band-pass filters
            survey_bands: all band-pass filters available for this survey
            tile_slen: dimension in pixels of a square tile
            tiles_to_crop: margin of tiles not to use for computing loss
            slack: Slack to use when matching locations for validation metrics.
            min_flux_threshold: Sources with a lower flux will not be considered when computing loss
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
            input_transform_params: used for determining what channels to use as input (e.g.
                deconvolution, concatenate PSF parameters, z-score inputs, etc.)
            do_data_augmentation: used for determining whether or not do data augmentation
            compile_model: compile model for potential performance improvements
        """
        super().__init__()
        self.save_hyperparameters()

        self.STAR_FLUX_NAMES = [f"star_flux_{bnd}" for bnd in survey_bands]  # ordered by BANDS
        self.GAL_FLUX_NAMES = [f"galaxy_flux_{bnd}" for bnd in survey_bands]  # ordered by BANDS

        self.bands = bands
        self.survey_bands = survey_bands
        self.n_bands = len(self.bands)
        self.min_flux_threshold = min_flux_threshold
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.input_transform_params = input_transform_params
        self.do_data_augmentation = do_data_augmentation

        transform_enabled = (
            "log_transform" in self.input_transform_params
            or "rolling_z_score" in self.input_transform_params
        )
        if self.n_bands > 1 and not transform_enabled:
            warnings.warn(
                "For multiband data, it is recommended that either"
                + " log transform or rolling z-score is enabled."
            )

        self.tile_slen = tile_slen

        # number of distributional parameters used to characterize each source
        self.n_params_per_source = sum(param.dim for param in self.dist_param_groups.values())

        # a hack to get the right number of outputs from yolo
        architecture["nc"] = self.n_params_per_source - 5
        arch_dict = OmegaConf.to_container(architecture)

        num_channels = len(self.bands)
        num_features_per_band = self._get_num_features()
        self.model = Backbone(cfg=arch_dict, ch=num_channels, n_imgs=num_features_per_band)
        if compile_model:
            self.model = torch.compile(self.model)
        self.tiles_to_crop = tiles_to_crop

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
        for star_flux in self.STAR_FLUX_NAMES:
            d[star_flux] = UnconstrainedLogNormal()
        for gal_flux in self.GAL_FLUX_NAMES:
            d[gal_flux] = UnconstrainedLogNormal()
        return d

    def _get_num_features(self):
        """Determine number of input channels for model based on desired input transforms."""
        num_features_per_band = 2  # img + bg
        if self.input_transform_params.get("use_deconv_channel"):
            num_features_per_band += 1
        if self.input_transform_params.get("concat_psf_params"):
            num_features_per_band += 6
        if self.input_transform_params.get("clahe"):
            num_features_per_band += 1
        return num_features_per_band

    def get_input_tensor(self, batch):
        """Extracts data from batch and concatenates into a single tensor to be input into model.

        By default, only the image and background are used. Other input transforms can be specified
        in self.input_transform_params. Supported options are:
            use_deconv_channel: add channel for image deconvolved with PSF
            concat_psf_params: add each PSF parameter as a channel
            rolling_z_score: rolling z-score both the images and background
            log_transform: apply pixel-wise natural logarithm to background-subtracted image.

        Args:
            batch: input batch (as dictionary)

        Returns:
            Tensor: b x c x 2 x h x w tensor, where the number of input channels `c` is based on the
                input transformations to use
        """
        input_bands = batch["images"].shape[1]
        if input_bands < self.n_bands:
            warnings.warn(
                f"Expected at least {self.n_bands} bands in the input but found only {input_bands}"
            )
        imgs = batch["images"][:, self.bands].unsqueeze(2)  # add extra dim for 5d input
        bgs = batch["background"][:, self.bands].unsqueeze(2)
        inputs = [imgs, bgs]

        if self.input_transform_params.get("use_deconv_channel"):
            assert (
                "deconvolution" in batch
            ), "use_deconv_channel specified but deconvolution not present in data"
            inputs.append(batch["deconvolution"][:, self.bands].unsqueeze(2))
        if self.input_transform_params.get("concat_psf_params"):
            assert (
                "psf_params" in batch
            ), "concat_psf_params specified but psf params not present in data"
            n, c, i, h, w = imgs.shape
            psf_params = batch["psf_params"][:, self.bands]
            inputs.append(psf_params.view(n, c, 6 * i, 1, 1).expand(n, c, 6 * i, h, w))
        if self.input_transform_params.get("log_transform"):
            # untransformed background
            inputs[0] = log_transform(torch.clamp(inputs[0] - inputs[1], min=1))
        elif self.input_transform_params.get("rolling_z_score"):
            inputs[0] = rolling_z_score(inputs[0], 9, 200, 4)
            inputs[1] = rolling_z_score(inputs[1], 9, 200, 4)
        return torch.cat(inputs, dim=2)

    def encode_batch(self, batch):
        # get input tensor from batch with specified channels and transforms
        inputs = self.get_input_tensor(batch)

        # setting this to true every time is a hack to make yolo DetectionModel
        # give us output of the right dimension
        self.model.model[-1].training = True

        assert inputs.size(3) % 16 == 0, "image dims must be multiples of 16"
        assert inputs.size(4) % 16 == 0, "image dims must be multiples of 16"

        output = self.model(inputs)
        # there's an extra dimension for channel that is always a singleton
        output4d = rearrange(output[0], "b 1 ht wt pps -> b ht wt pps")

        ttc = self.tiles_to_crop
        if ttc > 0:
            output4d = output4d[:, ttc:-ttc, ttc:-ttc, :]

        split_sizes = [v.dim for v in self.dist_param_groups.values()]
        dist_params_split = torch.split(output4d, split_sizes, 3)
        names = self.dist_param_groups.keys()
        pred = dict(zip(names, dist_params_split))

        for k, v in pred.items():
            pred[k] = self.dist_param_groups[k].get_dist(v)

        return pred

    def variational_mode(
        self, pred: Dict[str, Tensor], return_full: Optional[bool] = True
    ) -> Union[FullCatalog, TileCatalog]:
        """Compute the mode of the variational distribution.

        Args:
            pred (Dict[str, Tensor]): model predictions
            return_full (bool, optional): Returns a FullCatalog if true, otherwise returns a
                TileCatalog. Defaults to True.

        Returns:
            Union[FullCatalog, TileCatalog]: Catalog based on predictions.
        """
        # the mean would be better at minimizing squared error...should we return that instead?
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

        est_tile_catalog = TileCatalog(self.tile_slen, est_catalog_dict)
        return est_tile_catalog if not return_full else est_tile_catalog.to_full_params()

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
        pred = self.encode_batch(batch)
        true_tile_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])
        true_tile_cat = true_tile_cat.symmetric_crop(self.tiles_to_crop)

        # Filter by detectable sources and brightest source per tile
        target_cat = true_tile_cat
        if true_tile_cat.max_sources > 1:
            target_cat = target_cat.get_brightest_source_per_tile(band=2)
        if self.min_flux_threshold > 0:
            target_cat = target_cat.filter_tile_catalog_by_flux(min_flux=self.min_flux_threshold)

        loss_dict = self._get_loss(pred, target_cat)
        est_tile_cat = self.variational_mode(pred, return_full=False)  # get tile cat for metrics

        # log all losses
        for k, v in loss_dict.items():
            self.log("{}/{}".format(logging_name, k), v, batch_size=batch_size)

        # log all metrics
        if log_metrics:
            metrics = self.metrics(target_cat, est_tile_cat)
            for k, v in metrics.items():
                self.log("{}/{}".format(logging_name, k), v, batch_size=batch_size)

        # log a grid of figures to the tensorboard
        if plot_images:
            batch_size = len(batch["images"])
            n_samples = min(int(math.sqrt(batch_size)) ** 2, 16)
            nrows = int(n_samples**0.5)  # for figure

            target_full_cat = target_cat.to_full_params()
            est_full_cat = est_tile_cat.to_full_params()
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

        return loss_dict["loss"]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        if self.do_data_augmentation:
            imgs = batch["images"][:, self.bands].unsqueeze(2)  # add extra dim for 5d input
            bgs = batch["background"][:, self.bands].unsqueeze(2)
            aug_input_images = [imgs, bgs]
            if self.input_transform_params.get("use_deconv_channel"):
                assert (
                    "deconvolution" in batch
                ), "use_deconv_channel specified but deconvolution not present in data"
                aug_input_images.append(batch["background"][:, self.bands].unsqueeze(2))
            aug_input_image = torch.cat(aug_input_images, dim=2)

            aug_output_image, tile = augment_data(batch["tile_catalog"], aug_input_image)
            batch["images"] = aug_output_image[:, :, 0, :, :]
            batch["background"] = aug_output_image[:, :, 1, :, :]
            batch["tile_catalog"] = tile
            if self.input_transform_params.get("use_deconv_channel"):
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
            pred = self.encode_batch(batch)
            est_cat = self.variational_mode(pred, return_full=False)
        return {
            "est_cat": est_cat,
            "images": batch["images"],
            "background": batch["background"],
            "pred": pred,
        }
