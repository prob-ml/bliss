import math
import warnings
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from yolov5.models.yolo import DetectionModel

from bliss.catalog import FullCatalog, RegionCatalog, RegionType, SourceType, TileCatalog
from bliss.metrics import BlissMetrics, MetricsMode
from bliss.plotting import plot_detections
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS
from bliss.transforms import z_score
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

    STAR_FLUX_NAMES = [f"star_flux_{bnd}" for bnd in SDSS.BANDS]  # ordered by BANDS
    GAL_FLUX_NAMES = [f"galaxy_flux_{bnd}" for bnd in SDSS.BANDS]  # ordered by BANDS
    GALSIM_NAMES = ["disk_frac", "beta_radians", "disk_q", "a_d", "bulge_q", "a_b"]

    def __init__(
        self,
        architecture: DictConfig,
        bands: list,
        tile_slen: int,
        tiles_to_crop: int,
        slack: float = 1.0,
        min_flux_threshold: float = 0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        input_transform_params: Optional[dict] = None,
    ):
        """Initializes DetectionEncoder.

        Args:
            architecture: yaml to specifying the encoder network architecture
            bands: specified band-pass filters
            tile_slen: dimension in pixels of a square tile
            tiles_to_crop: margin of tiles not to use for computing loss
            slack: Slack to use when matching locations for validation metrics.
            min_flux_threshold: Sources with a lower flux will not be considered when computing loss
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
            input_transform_params: used for determining what channels to use as input (e.g.
                deconvolution, concatenate PSF parameters, z-score inputs, etc.)
        """
        super().__init__()
        self.save_hyperparameters()

        self.bands = bands
        self.n_bands = len(self.bands)
        self.min_flux_threshold = min_flux_threshold
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.input_transform_params = input_transform_params

        if not self.input_transform_params.get("z_score") and self.n_bands > 1:
            warnings.warn("Performing multi-band encoding without z-scoring.")

        self.tile_slen = tile_slen

        # number of distributional parameters used to characterize each source
        self.n_params_per_source = sum(param.dim for param in self.dist_param_groups.values())

        # a hack to get the right number of outputs from yolo
        architecture["nc"] = self.n_params_per_source - 5
        arch_dict = OmegaConf.to_container(architecture)

        num_channels = self._get_num_input_channels()
        self.model = DetectionModel(cfg=arch_dict, ch=num_channels)
        self.tiles_to_crop = tiles_to_crop

        # metrics
        self.metrics = BlissMetrics(mode=MetricsMode.TILE, slack=slack)

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

    def _get_num_input_channels(self):
        """Determine number of input channels for model based on desired input transforms."""
        num_channels = 2  # image + background
        if self.input_transform_params.get("use_deconv_channel"):
            num_channels += 1
        if self.input_transform_params.get("concat_psf_params"):
            num_channels += 6
        num_channels *= self.n_bands  # multi-band support
        return num_channels

    def get_input_tensor(self, batch):
        """Extracts data from batch and concatenates into a single tensor to be input into model.

        By default, only the image and background are used. Other input transforms can be specified
        in self.input_transform_params. Supported options are:
            use_deconv_channel: add channel for image deconvolved with PSF
            concat_psf_params: add each PSF parameter as a channel
            z_score: z-score both the images and background

        Args:
            batch: input batch (as dictionary)

        Returns:
            Tensor: b x c x h x w tensor, where the number of input channels `c` is based on the
                input transformations to use
        """
        input_bands = batch["images"].shape[1]
        if input_bands < self.n_bands:
            warnings.warn(
                f"Expected at least {self.n_bands} bands in the input but found only {input_bands}"
            )
        imgs = batch["images"][:, self.bands]
        bgs = batch["background"][:, self.bands]
        inputs = [imgs, bgs]

        if self.input_transform_params.get("use_deconv_channel"):
            assert (
                "deconvolution" in batch
            ), "use_deconv_channel specified but deconvolution not present in data"
            inputs.append(batch["deconvolution"][:, self.bands])
        if self.input_transform_params.get("concat_psf_params"):
            assert (
                "psf_params" in batch
            ), "concat_psf_params specified but psf params not present in data"
            n, c, h, w = imgs.shape
            psf_params = batch["psf_params"][:, self.bands]
            inputs.append(psf_params.view(n, 6 * c, 1, 1).expand(n, 6 * c, h, w))
        if self.input_transform_params.get("z_score"):
            assert (
                batch["background"][0, 0].std() > 0
            ), "Constant backgrounds not supported for multi-band encoding"
            inputs[0] = z_score(inputs[0])
            inputs[1] = z_score(inputs[1])
        return torch.cat(inputs, dim=1)

    def encode_batch(self, batch):
        # get input tensor from batch with specified channels and transforms
        inputs = self.get_input_tensor(batch)

        # setting this to true every time is a hack to make yolo DetectionModel
        # give us output of the right dimension
        self.model.model[-1].training = True

        assert inputs.size(2) % 16 == 0, "image dims must be multiples of 16"
        assert inputs.size(3) % 16 == 0, "image dims must be multiples of 16"

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

    # region Loss Utility Functions
    def _average_loss(self, loss, mask):
        """Return the average loss in regions specified by mask."""
        if mask.sum() == 0:
            return 0
        return loss.sum() / mask.sum()

    def _get_masked_param(self, param, mask, shape):
        """Get param in masked regions. `shape` controls output shape for different region types."""
        b, nth, ntw, d = *shape, param.shape[-1]
        if len(param.shape) == 3:
            mask = repeat(mask, "nth ntw -> b nth ntw", b=b)
            return param[mask].reshape(shape)

        param = param.squeeze(-2)  # remove max_sources dim
        mask = repeat(mask, "r c -> b r c d", b=b, d=d)
        param = param[mask]
        masked_params = rearrange(param, "(b nth ntw d) -> b nth ntw d", b=b, nth=nth, ntw=ntw, d=d)

        return masked_params.squeeze(-1) if d == 1 else masked_params

    def _get_interior_param(self, cat: RegionCatalog, param_name: str):
        """Get param in interior regions."""
        out_shape = (cat.batch_size, cat.nth, cat.ntw)
        param = cat.get_interior_locs_in_tile() if param_name == "locs" else cat[param_name]
        return self._get_masked_param(param, cat.interior_mask, out_shape)

    def _get_vertical_boundary_param(self, cat: RegionCatalog, param_name: str):
        """Get param in vertical boundary regions."""
        out_shape = (cat.batch_size, cat.nth, cat.ntw - 1)
        if param_name == "locs":
            locs_left, locs_right = cat.get_vertical_boundary_locs_in_tiles()
            locs_left = self._get_masked_param(locs_left, cat.vertical_boundary_mask, out_shape)
            locs_right = self._get_masked_param(locs_right, cat.vertical_boundary_mask, out_shape)
            return locs_left, locs_right

        return self._get_masked_param(cat[param_name], cat.vertical_boundary_mask, out_shape)

    def _get_horizontal_boundary_param(self, cat: RegionCatalog, param_name: str):
        """Get param in horizontal boundary regions."""
        out_shape = (cat.batch_size, cat.nth - 1, cat.ntw)
        if param_name == "locs":
            locs_up, locs_down = cat.get_horizontal_boundary_locs_in_tiles()
            locs_up = self._get_masked_param(locs_up, cat.vertical_boundary_mask, out_shape)
            locs_down = self._get_masked_param(locs_down, cat.vertical_boundary_mask, out_shape)
            return locs_up, locs_down

        return self._get_masked_param(cat[param_name], cat.horizontal_boundary_mask, out_shape)

    def _get_aux_vertical(self, pred, cat):
        """Get auxiliary variables in vertical boundary regions."""
        idx_v = repeat(
            cat.vertical_boundary_mask, "nth ntw -> b nth ntw", b=cat.batch_size
        ).nonzero(as_tuple=True)
        idx_vi = (idx_v[0], idx_v[1] // 2, idx_v[2] // 2)
        idx_vj = (idx_v[0], idx_v[1] // 2, (idx_v[2] + 1) // 2)

        probs = pred["on_prob"].probs[..., 1]  # prob of yes
        aux_vars = torch.zeros(cat.batch_size, cat.nth, cat.ntw - 1, device=cat.device)
        aux_vars[idx_vi] = probs[idx_vi] / (probs[idx_vi] + probs[idx_vj])
        return aux_vars

    def _get_aux_horizontal(self, pred, cat):
        """Get auxiliary variables in horizontal boundary regions."""
        idx_h = repeat(
            cat.horizontal_boundary_mask, "nth ntw -> b nth ntw", b=cat.batch_size
        ).nonzero(as_tuple=True)
        idx_hi = (idx_h[0], idx_h[1] // 2, idx_h[2] // 2)
        idx_hj = (idx_h[0], (idx_h[1] + 1) // 2, idx_h[2] // 2)

        probs = pred["on_prob"].probs[..., 1]  # prob of yes
        aux_vars = torch.zeros(cat.batch_size, cat.nth - 1, cat.ntw, device=cat.device)
        aux_vars[idx_hi] = probs[idx_hi] / (probs[idx_hi] + probs[idx_hj])
        return aux_vars

    # endregion

    # region Main Loss Functions
    def _get_loss_interior(self, pred: Dict[str, Distribution], cat: RegionCatalog):
        """Compute loss in interior regions.

        Args:
            pred (Dict[str, Distribution]): predicted distributions to evaluate
            cat (RegionCatalog): true catalog

        Returns:
            Dict: dictionary of loss for each component and overall loss
        """
        loss, loss_components = 0, {}

        # counter_loss
        n_sources = self._get_interior_param(cat, "n_sources")
        on_probs = pred["on_prob"].log_prob(n_sources)
        interior_ub = torch.ones_like(pred["loc"].mean) - (cat.overlap_slen / cat.tile_slen)
        interior_lb = torch.zeros_like(pred["loc"].mean) + (cat.overlap_slen / cat.tile_slen)
        prob_in_interior = pred["loc"].cdf(interior_ub) - pred["loc"].cdf(interior_lb)

        counter_loss = -((n_sources == 0) * torch.log(1 - on_probs.exp() * prob_in_interior))
        counter_loss -= (n_sources > 0) * pred["on_prob"].log_prob(n_sources)
        loss += counter_loss
        loss_components["counter_loss"] = counter_loss.mean()

        # location loss
        locs = self._get_interior_param(cat, "locs")
        locs_loss = (-pred["loc"].log_prob(locs)) * n_sources
        loss += locs_loss
        loss_components["locs_loss"] = self._average_loss(locs_loss, n_sources)

        # star/galaxy classification loss
        gal_bools = self._get_interior_param(cat, "galaxy_bools")
        binary_loss = (-pred["galaxy_prob"].log_prob(gal_bools)) * n_sources
        loss += binary_loss
        loss_components["binary_loss"] = self._average_loss(binary_loss, n_sources)

        # flux losses
        star_bools = self._get_interior_param(cat, "star_bools")
        star_fluxes = self._get_interior_param(cat, "star_fluxes")
        galaxy_fluxes = self._get_interior_param(cat, "galaxy_fluxes")

        # only compute loss over bands we're using
        star_bands = [self.STAR_FLUX_NAMES[band] for band in self.bands]
        gal_bands = [self.GAL_FLUX_NAMES[band] for band in self.bands]
        for band, star_name, gal_name in zip(self.bands, star_bands, gal_bands):
            # star flux loss
            star_flux_loss = -pred[star_name].log_prob(star_fluxes[..., band] + 1e-9) * star_bools
            loss += star_flux_loss
            loss_components[star_name] = self._average_loss(star_flux_loss, star_bools)

            # galaxy flux loss
            gal_flux_loss = -pred[gal_name].log_prob(galaxy_fluxes[..., band] + 1e-9) * gal_bools
            loss += gal_flux_loss
            loss_components[gal_name] = self._average_loss(gal_flux_loss, gal_bools)

        # galaxy properties loss
        galsim_true_vals = self._get_interior_param(cat, "galaxy_params")
        for i, param_name in enumerate(self.GALSIM_NAMES):
            galsim_pn = f"galsim_{param_name}"
            gal_param_loss = -pred[galsim_pn].log_prob(galsim_true_vals[..., i] + 1e-9) * gal_bools
            loss += gal_param_loss
            loss_components[galsim_pn] = self._average_loss(gal_param_loss, gal_bools)

        loss_by_region = torch.zeros(cat.batch_size, cat.n_rows, cat.n_cols, device=cat.device)
        loss_by_region[:, ::2, ::2] = loss
        loss_components["loss_by_region"] = loss_by_region
        return loss_components

    def _get_param_loss_boundary(self, dist, val_i, val_j, cat, aux_vars, mask, bdry_type):
        """Compute the loss for a single param in boundary regions.

        The loss is computed by evaluating the value of the param in the tiles to the left and
        right of the boundary (or above and below). We use logsumexp for numerical stability.

        Args:
            dist: the distribution to evaluate
            val_i: the value to evaluate at in the tile on one side of the boundary
            val_j: the value to evaluate at in the tile on the other side of the boundary
            cat: true catalog
            aux_vars: the auxiliary weights of the left and right tiles
            mask: the mask to apply to the final loss
            bdry_type: type of boundary, RegionType.BOUNDARY_VERTICAL or
                RegionType.BOUNDARY_HORIZONTAL

        Returns:
            Tensor: a tensor of the loss for this param in each vertical boundary (and 0s in all
                other regions)
        """
        shape = list(val_i.shape)
        c = 1e-12 if isinstance(dist, torch.distributions.LogNormal) else 0  # ensure val in support

        # get probs in left/right tile for vertical, above/below for horizontal
        if bdry_type == RegionType.BOUNDARY_VERTICAL:
            shape[2] = 1
            col = torch.zeros(*shape, device=cat.device)
            log_prob_i = dist.log_prob(torch.cat((val_i, col), dim=2) + c)[..., :-1]
            log_prob_j = dist.log_prob(torch.cat((col, val_j), dim=2) + c)[..., 1:]
        else:
            shape[1] = 1
            row = torch.zeros(shape, device=cat.device)
            log_prob_i = dist.log_prob(torch.cat((val_i, row), dim=1) + c)[:, :-1]
            log_prob_j = dist.log_prob(torch.cat((row, val_j), dim=1) + c)[:, 1:]

        # evaluate prob using logsumexp for stability
        log_prob_i += torch.log(aux_vars)
        log_prob_j += torch.log(1 - aux_vars)
        prob = -torch.logsumexp(torch.stack((log_prob_i, log_prob_j), dim=3), dim=3)

        # construct loss array and add values to appropriate regions
        loss = torch.zeros(cat.batch_size, cat.n_rows, cat.n_cols, device=cat.device)
        if bdry_type == RegionType.BOUNDARY_VERTICAL:
            loss[:, ::2, 1::2] = prob
        else:
            loss[:, 1::2, ::2] = prob
        return loss * mask

    def _get_loss_boundary(
        self, pred: Dict[str, Distribution], cat: RegionCatalog, bdry_type: RegionType
    ):
        """Compute loss in boundary regions.

        Args:
            pred (Dict[str, Distribution]): predicted distributions to evaluate
            cat (RegionCatalog): true catalog
            bdry_type (RegionType): which regions to evaluate, either RegionType.BOUNDARY_VERTICAL
                or RegionType.BOUNDARY_HORIZONTAL

        Returns:
            Dict: dictionary of loss for each component and overall loss
        """
        assert bdry_type in {RegionType.BOUNDARY_VERTICAL, RegionType.BOUNDARY_HORIZONTAL}
        loss, loss_components = 0, {}
        if bdry_type == RegionType.BOUNDARY_VERTICAL:
            on_mask = cat.vertical_boundary_mask
            aux_vars = self._get_aux_vertical(pred, cat)
            get_param = self._get_vertical_boundary_param
        else:
            on_mask = cat.horizontal_boundary_mask
            aux_vars = self._get_aux_horizontal(pred, cat)
            get_param = self._get_horizontal_boundary_param

        # counter_loss
        n_sources = get_param(cat, "n_sources")
        counter_loss = self._get_param_loss_boundary(
            pred["on_prob"], n_sources, n_sources, cat, aux_vars, on_mask, bdry_type
        )
        loss += counter_loss
        loss_components["counter_loss"] = self._average_loss(
            counter_loss, cat.vertical_boundary_mask
        )

        # location loss
        locs_left, locs_right = get_param(cat, "locs")
        on_mask = on_mask * (cat.n_sources > 0)  # update mask to filter on sources
        locs_loss = self._get_param_loss_boundary(
            pred["loc"], locs_left, locs_right, cat, aux_vars, on_mask, bdry_type
        )
        loss += locs_loss
        loss_components["locs_loss"] = self._average_loss(locs_loss, on_mask)

        # star/galaxy classification loss
        gal_bools = get_param(cat, "galaxy_bools")
        binary_loss = self._get_param_loss_boundary(
            pred["galaxy_prob"], gal_bools, gal_bools, cat, aux_vars, on_mask, bdry_type
        )
        loss += binary_loss
        loss_components["binary_loss"] = self._average_loss(binary_loss, on_mask)

        # flux losses
        star_fluxes = get_param(cat, "star_fluxes")
        galaxy_fluxes = get_param(cat, "galaxy_fluxes")
        star_mask = on_mask * cat.star_bools[..., 0, 0]
        gal_mask = on_mask * cat.galaxy_bools[..., 0, 0]

        for i, (star_name, gal_name) in enumerate(zip(self.STAR_FLUX_NAMES, self.GAL_FLUX_NAMES)):
            if i not in self.bands:  # only compute loss over bands we're using
                continue
            # star flux loss
            fluxes = star_fluxes[..., i]
            star_flux_loss = self._get_param_loss_boundary(
                pred[star_name], fluxes, fluxes, cat, aux_vars, star_mask, bdry_type
            )
            loss += star_flux_loss
            loss_components[star_name] = self._average_loss(star_flux_loss, star_mask)

            # galaxy flux loss
            fluxes = galaxy_fluxes[..., i]
            galaxy_flux_loss = self._get_param_loss_boundary(
                pred[gal_name], fluxes, fluxes, cat, aux_vars, gal_mask, bdry_type
            )
            loss += galaxy_flux_loss
            loss_components[gal_name] = self._average_loss(galaxy_flux_loss, gal_mask)

        # galaxy properties loss
        galaxy_params = get_param(cat, "galaxy_params")
        for i, param_name in enumerate(self.GALSIM_NAMES):
            galsim_pn = f"galsim_{param_name}"
            gal_param = galaxy_params[..., i]
            gal_param_loss = self._get_param_loss_boundary(
                pred[galsim_pn], gal_param, gal_param, cat, aux_vars, gal_mask, bdry_type
            )
            loss += gal_param_loss
            loss_components[galsim_pn] = self._average_loss(gal_param_loss, gal_mask)

        loss_components["loss_by_region"] = loss
        return loss_components

    def _get_loss_corner(self, pred: Dict[str, Distribution], cat: RegionCatalog):
        # TODO: implement this
        pass

    def _get_loss(self, pred: Dict[str, Distribution], true_cat: RegionCatalog):
        """Compute loss over the catalog."""
        # interior
        loss_components = self._get_loss_interior(pred, true_cat)
        # vertical boundary
        loss_v_boundary = self._get_loss_boundary(pred, true_cat, RegionType.BOUNDARY_VERTICAL)
        # horizontal boundary
        loss_h_boundary = self._get_loss_boundary(pred, true_cat, RegionType.BOUNDARY_HORIZONTAL)
        # TODO: corners

        # sum loss for all regions
        for key in loss_components:
            loss_components[key] += loss_v_boundary[key]
            loss_components[key] += loss_h_boundary[key]
            # TODO: corner

        loss_components["loss"] = loss_components.pop("loss_by_region").mean()
        return loss_components

    # endregion

    # region Lightning Functions
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
                batch["images"], target_full_cat, est_full_cat, nrows, wrong_idx, margin_px
            )
            title_root = f"Epoch:{self.current_epoch}/"
            title = f"{title_root}{logging_name} images"
            if self.logger:
                self.logger.experiment.add_figure(title, fig)
            plt.close(fig)

        return loss_dict["loss"]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        return self._generic_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        # only plot images on the first batch of epoch
        plot_images = batch_idx == 0
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

    # endregion
