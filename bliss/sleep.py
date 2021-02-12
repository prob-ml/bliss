"""Summary for sleep.py

This module contains SleepPhase class, which implements the sleep-phase traning using
pytorch-lightning framework. Users should use this class to construct the sleep-phase
model.

"""

import math
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pytorch_lightning as pl

import torch
from torch.nn import CrossEntropyLoss
from torch.distributions import Normal
from torch.optim import Adam

from . import plotting
from .models import encoder, decoder, galaxy_net
from .models.encoder import get_star_bool, get_full_params
from .metrics import eval_error_on_batch

plt.switch_backend("Agg")


def sort_locs(locs):
    # sort according to x location
    assert len(locs.shape) == 2
    indx_sort = locs[:, 0].sort()[1]
    return locs[indx_sort, :], indx_sort


def _get_log_probs_all_perms(
    locs_log_probs_all,
    star_params_log_probs_all,
    prob_galaxy,
    true_galaxy_bool,
    is_on_array,
):
    # get log-probability under every possible matching of estimated source to true source
    n_ptiles = star_params_log_probs_all.size(0)
    max_detections = star_params_log_probs_all.size(-1)

    n_permutations = math.factorial(max_detections)
    locs_log_probs_all_perm = torch.zeros(
        n_ptiles, n_permutations, device=locs_log_probs_all.device
    )
    star_params_log_probs_all_perm = locs_log_probs_all_perm.clone()
    galaxy_bool_log_probs_all_perm = locs_log_probs_all_perm.clone()

    for i, perm in enumerate(permutations(range(max_detections))):
        # note that we multiply is_on_array, we only evaluate the loss if the source is on.
        locs_log_probs_all_perm[:, i] = (
            locs_log_probs_all[:, perm].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)

        # if star, evaluate the star parameters,
        # hence the multiplication by (1 - true_galaxy_bool)
        # the diagonal is a clever way of selecting the elements of each permutation (first index
        # of mean/var with second index of true_param etc.)
        star_params_log_probs_all_perm[:, i] = (
            star_params_log_probs_all[:, perm].diagonal(dim1=1, dim2=2)
            * is_on_array
            * (1 - true_galaxy_bool)
        ).sum(1)

        _prob_galaxy = prob_galaxy[:, perm]
        galaxy_bool_loss = true_galaxy_bool * torch.log(_prob_galaxy)
        galaxy_bool_loss += (1 - true_galaxy_bool) * torch.log(1 - _prob_galaxy)
        galaxy_bool_log_probs_all_perm[:, i] = (galaxy_bool_loss * is_on_array).sum(1)

    return (
        locs_log_probs_all_perm,
        star_params_log_probs_all_perm,
        galaxy_bool_log_probs_all_perm,
    )


def _get_min_perm_loss(
    locs_log_probs_all,
    star_params_log_probs_all,
    prob_galaxy,
    true_galaxy_bool,
    is_on_array,
):
    # get log-probability under every possible matching of estimated star to true star
    (
        locs_log_probs_all_perm,
        star_params_log_probs_all_perm,
        galaxy_bool_log_probs_all_perm,
    ) = _get_log_probs_all_perms(
        locs_log_probs_all,
        star_params_log_probs_all,
        prob_galaxy,
        true_galaxy_bool,
        is_on_array,
    )

    # find the permutation that minimizes the location losses
    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim=1)

    # get the star & galaxy losses according to the found permutation
    _indx = indx.unsqueeze(1)
    star_params_loss = -torch.gather(star_params_log_probs_all_perm, 1, _indx).squeeze()
    galaxy_bool_loss = -torch.gather(galaxy_bool_log_probs_all_perm, 1, _indx).squeeze()
    return locs_loss, star_params_loss, galaxy_bool_loss


def _get_params_logprob_all_combs(true_params, param_mean, param_logvar):
    # return shape (n_ptiles x max_detections x max_detections)
    assert true_params.shape == param_mean.shape == param_logvar.shape

    n_ptiles = true_params.size(0)
    max_detections = true_params.size(1)

    # view to evaluate all combinations of log_prob.
    _true_params = true_params.view(n_ptiles, 1, max_detections, -1)
    _param_mean = param_mean.view(n_ptiles, max_detections, 1, -1)
    _param_logvar = param_logvar.view(n_ptiles, max_detections, 1, -1)

    _sd = (_param_logvar.exp() + 1e-5).sqrt()
    param_log_probs_all = Normal(_param_mean, _sd).log_prob(_true_params).sum(dim=3)
    return param_log_probs_all


class SleepPhase(pl.LightningModule):
    """Summary line.

    Implementation of sleep-phase training using pytorch-lightning framework.

    Args:
        cfg (DictConfig): OmegaConf configuration from YAML files

    Example:
        In python script, set up the sleep-phase model and pytorch-lightning trainer::

            import pytorch_lightning as pl
            from bliss.sleep import SleepPhase


            model = SleepPhase(cfg)
            trainer = pl.Trainer()
            trainer.fit(model, data=dataset)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.image_encoder = encoder.ImageEncoder(**cfg.model.encoder.params)
        self.image_decoder = decoder.ImageDecoder(**cfg.model.decoder.params)
        self.image_decoder.requires_grad_(False)

        self.plotting: bool = cfg.training.plotting

        # consistency
        assert self.image_decoder.tile_slen == self.image_encoder.tile_slen
        assert self.image_decoder.border_padding == self.image_encoder.border_padding
        assert self.image_encoder.max_detections <= self.image_decoder.max_sources

        self.use_galaxy_encoder = cfg.model.use_galaxy_encoder
        self.galaxy_encoder = None
        if self.use_galaxy_encoder:
            # NOTE: We crop and center each padded tile before passing it on to the galaxy_encoder
            #       assume that crop_slen = 2*tile_slen (on each side)
            # TODO: for now only, 1 galaxy per tile is supported. Even though multiple stars per
            #       tile should work but there is no easy way to enforce this.
            self.galaxy_encoder = galaxy_net.CenteredGalaxyEncoder(
                **cfg.model.galaxy_encoder.params
            )
            self.cropped_slen = (
                self.image_encoder.ptile_slen - 4 * self.image_encoder.tile_slen
            )
            assert self.cropped_slen >= 20, "Cropped slen not reasonable"
            assert self.galaxy_encoder.slen == self.cropped_slen
            assert self.galaxy_encoder.latent_dim == self.image_decoder.n_galaxy_params
            assert self.image_decoder.max_sources == 1, "1 galaxy per tile is supported"
            assert self.image_encoder.max_detections == 1

    def forward_galaxy(self, image_ptiles, tile_locs):
        n_ptiles = image_ptiles.shape[0]

        # in each padded tile we need to center the corresponding galaxy
        _tile_locs = tile_locs.reshape(n_ptiles, self.image_decoder.max_sources, 2)
        centered_ptiles = self.image_encoder.center_ptiles(image_ptiles, _tile_locs)
        assert centered_ptiles.shape[-1] == self.cropped_slen

        # remove background before encoding
        background_values = self.image_decoder.background.mean((1, 2))
        ptile_background = torch.zeros(
            1,
            self.image_encoder.n_bands,
            self.cropped_slen,
            self.cropped_slen,
            device=centered_ptiles.device,
        )
        for b in range(ptile_background.shape[1]):
            ptile_background[:, b] = background_values[b]
        centered_ptiles -= ptile_background

        # TODO: Should we zero out tiles without galaxies during training?
        # we can assume there is one galaxy per_tile and encode each tile independently.
        encoding = self.galaxy_encoder.forward(centered_ptiles)
        galaxy_param_mean, galaxy_param_var = encoding
        assert galaxy_param_mean.shape[0] == n_ptiles
        assert galaxy_param_var.shape[0] == n_ptiles

        return galaxy_param_mean, galaxy_param_var

    def forward(self, image_ptiles, n_sources):
        raise NotImplementedError()

    def tile_map_estimate(self, batch):
        # NOTE: batch is per tile since it comes from image_decoder
        images = batch["images"]
        tile_galaxy_params = batch["galaxy_params"]
        tile_params = self.image_encoder.tile_map_estimate(images)

        if self.use_galaxy_encoder:
            batch_size = images.shape[0]
            max_detections = 1
            tile_locs = tile_params["locs"].reshape(-1, max_detections, 2)
            image_ptiles = self.image_encoder.get_images_in_tiles(images)
            tile_galaxy_params, _ = self.forward_galaxy(image_ptiles, tile_locs)
            n_galaxy_params = tile_galaxy_params.shape[-1]
            tile_galaxy_params = tile_galaxy_params.reshape(
                batch_size,
                -1,
                max_detections,
                n_galaxy_params,
            )

        # TODO: True galaxy params are not necessarily consistent with MAP estimated
        # need to do some matching to ensure correctness of residual images?
        # maybe doesn't matter because only care about detection if not estimating
        # galaxy_parameters.
        max_sources = tile_params["locs"].shape[2]
        tile_galaxy_params = tile_galaxy_params[:, :, :max_sources].contiguous()
        tile_est = {**tile_params, "galaxy_params": tile_galaxy_params}
        return tile_est

    def get_galaxy_loss(self, batch):
        images = batch["images"]
        batch_size = images.shape[0]
        # shape = (n_ptiles x band x ptile_slen x ptile_slen)
        image_ptiles = self.image_encoder.get_images_in_tiles(images)
        n_galaxy_params = self.image_decoder.n_galaxy_params
        galaxy_param_mean, galaxy_param_var = self.forward_galaxy(
            image_ptiles, batch["locs"]
        )

        galaxy_param_mean = galaxy_param_mean.view(batch_size, -1, 1, n_galaxy_params)
        galaxy_param_var = galaxy_param_var.view(batch_size, -1, 1, n_galaxy_params)

        # start calculating kl_qp loss.
        q_z = Normal(galaxy_param_mean, galaxy_param_var.sqrt())
        z = q_z.rsample()
        log_q_z = q_z.log_prob(z).sum((1, 2, 3))
        p_z = Normal(torch.zeros_like(z), torch.ones_like(z))
        log_p_z = p_z.log_prob(z).sum((1, 2, 3))

        # now draw a full reconstructed image.
        recon_mean, recon_var = self.image_decoder.render_images(
            batch["n_sources"],
            batch["locs"],
            batch["galaxy_bool"],
            z,
            batch["fluxes"],
            add_noise=False,
        )

        kl_z = log_q_z - log_p_z  # log q(z | x) - log p(z)
        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(images)
        recon_losses = recon_losses.view(batch_size, -1).sum(1)
        kl_qp = (recon_losses + kl_z).sum()

        return kl_qp

    def get_detection_loss(self, batch):
        """

        loc_mean shape = (n_ptiles x max_detections x 2)
        log_flux_mean shape = (n_ptiles x max_detections x n_bands)

        the *_logvar inputs should the same shape as their respective means
        the true_tile_* inputs, except for true_tile_is_on_array,
        should have same shape as their respective means, e.g.
        true_locs should have the same shape as loc_mean

        In true_locs, the off sources must have parameter value = 0

        true_is_on_array shape = (n_ptiles x max_detections)
            Indicates if sources is on (1) or off (0)

        true_galaxy_bool shape = (n_ptiles x max_detections x 1)
            indicating whether each source is a galaxy (1) or star (0)

        prob_galaxy shape = (n_ptiles x max_detections)
            are probabilities for each source to be a galaxy

        n_source_log_probs shape = (n_ptiles x (max_detections + 1))
            are log-probabilities for the number of sources (0, 1, ..., max_detections)

        """
        (
            images,
            true_tile_locs,
            true_tile_log_fluxes,
            true_tile_galaxy_bool,
            true_tile_n_sources,
        ) = (
            batch["images"],
            batch["locs"],
            batch["log_fluxes"],
            batch["galaxy_bool"],
            batch["n_sources"],
        )

        # some constants
        batch_size = images.shape[0]
        n_tiles_per_image = self.image_decoder.n_tiles_per_image
        n_ptiles = batch_size * n_tiles_per_image
        max_sources = self.image_encoder.max_detections
        n_bands = self.image_decoder.n_bands

        # clip decoder output since constraint is: max_detections <= max_sources (per tile)
        true_tile_locs = true_tile_locs[:, :, 0:max_sources]
        true_tile_log_fluxes = true_tile_log_fluxes[:, :, 0:max_sources]
        true_tile_galaxy_bool = true_tile_galaxy_bool[:, :, 0:max_sources]
        true_tile_n_sources = true_tile_n_sources.clamp(max=max_sources)

        # flatten so first dimension is ptile
        true_tile_locs = true_tile_locs.view(n_ptiles, max_sources, 2)
        true_tile_log_fluxes = true_tile_log_fluxes.view(n_ptiles, max_sources, n_bands)
        true_tile_galaxy_bool = true_tile_galaxy_bool.view(n_ptiles, max_sources)
        true_tile_n_sources = true_tile_n_sources.view(n_ptiles)
        true_tile_is_on_array = encoder.get_is_on_from_n_sources(
            true_tile_n_sources, max_sources
        )

        # extract image tiles
        image_ptiles = self.image_encoder.get_images_in_tiles(images)
        pred = self.image_encoder.forward(image_ptiles, true_tile_n_sources)

        # the loss for estimating the true number of sources
        n_source_log_probs = pred["n_source_log_probs"].view(n_ptiles, max_sources + 1)
        cross_entropy = CrossEntropyLoss(reduction="none").requires_grad_(False)
        counter_loss = cross_entropy(n_source_log_probs, true_tile_n_sources)

        # the following two functions computes the log-probability of parameters when
        # each estimated source i is matched with true source j for
        # i, j in {1, ..., max_detections}
        # *_log_probs_all have shape n_ptiles x max_detections x max_detections

        # enforce large error if source is off
        loc_mean, loc_logvar = pred["loc_mean"], pred["loc_logvar"]
        loc_mean = loc_mean + (true_tile_is_on_array == 0).float().unsqueeze(-1) * 1e16
        locs_log_probs_all = _get_params_logprob_all_combs(
            true_tile_locs, loc_mean, loc_logvar
        )
        star_params_log_probs_all = _get_params_logprob_all_combs(
            true_tile_log_fluxes, pred["log_flux_mean"], pred["log_flux_logvar"]
        )
        prob_galaxy = pred["prob_galaxy"].view(n_ptiles, max_sources)

        # inside _get_min_perm_loss is where the matching happens:
        # we construct a bijective map from each estimated source to each true source
        (locs_loss, star_params_loss, galaxy_bool_loss,) = _get_min_perm_loss(
            locs_log_probs_all,
            star_params_log_probs_all,
            prob_galaxy,
            true_tile_galaxy_bool,
            true_tile_is_on_array,
        )

        loss_vec = (
            locs_loss * (locs_loss.detach() < 1e6).float()
            + counter_loss
            + star_params_loss
            + galaxy_bool_loss
        )
        loss = loss_vec.mean()

        return (
            loss,
            counter_loss,
            locs_loss,
            star_params_loss,
            galaxy_bool_loss,
        )

    def configure_optimizers(self):
        params = self.hparams.optimizer.params
        opt = Adam(self.image_encoder.parameters(), **params)

        if self.use_galaxy_encoder:
            galaxy_opt = Adam(self.galaxy_encoder.parameters(), **params)
            opt = (opt, galaxy_opt)

        return opt

    def training_step(
        self, batch, batch_idx, optimizer_idx=0
    ):  # pylint: disable=unused-argument
        loss = 0.0

        if optimizer_idx == 0:  # image_encoder
            loss = self.get_detection_loss(batch)[0]
            self.log("train_detection_loss", loss)

        if optimizer_idx == 1:  # galaxy_encoder:
            loss = self.get_galaxy_loss(batch)
            self.log("train_galaxy_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        (
            detection_loss,
            counter_loss,
            locs_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = self.get_detection_loss(batch)

        self.log("val_loss", detection_loss)
        self.log("val_counter_loss", counter_loss.mean())
        self.log("val_locs_loss", locs_loss.mean())
        self.log("val_gal_bool_loss", galaxy_bool_loss.mean())
        self.log("val_star_params_loss", star_params_loss.mean())

        if self.use_galaxy_encoder:
            galaxy_loss = self.get_galaxy_loss(batch)
            self.log("val_galaxy_loss", galaxy_loss)

        # calculate metrics for this batch
        metrics = self.get_metrics(batch)
        self.log("val_acc_counts", metrics["counts_acc"])
        self.log("val_gal_counts", metrics["galaxy_counts_acc"])
        self.log("val_locs_mae", metrics["locs_mae"])
        self.log("val_star_fluxes_mae", metrics["star_fluxes_mae"])
        self.log("val_avg_tpr", metrics["avg_tpr"])
        self.log("val_avg_ppv", metrics["avg_ppv"])
        self.log("val_galaxy_params_mae", metrics["galaxy_params_mae"])
        self.log("val_image_fluxes_mae", metrics["image_fluxes_mae"])
        self.log("val_norm_pp_mae", metrics["norm_pp_mae"])
        return batch

    def validation_epoch_end(self, outputs):
        # NOTE: outputs is a list containing all validation step batches.
        if self.plotting and self.current_epoch > 1:
            self.make_plots(outputs[-1], kind="validation")

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        metrics = self.get_metrics(batch)
        self.log("acc_counts", metrics["counts_acc"])
        self.log("acc_gal_counts", metrics["galaxy_counts_acc"])
        self.log("locs_mae", metrics["locs_mae"])
        self.log("star_fluxes_mae", metrics["star_fluxes_mae"])
        self.log("avg_tpr", metrics["avg_tpr"])
        self.log("avg_ppv", metrics["avg_ppv"])
        self.log("galaxy_params_mae", metrics["galaxy_params_mae"])
        self.log("image_fluxes_mae", metrics["image_fluxes_mae"])
        self.log("norm_pp_mae", metrics["norm_pp_mae"])

        return batch

    def test_epoch_end(self, outputs):
        batch = outputs[-1]
        if self.plotting:
            self.make_plots(batch, kind="testing")

    def get_metrics(self, batch):
        # get images and properties
        exclude = {"images", "slen", "background"}
        true_images = batch["images"]
        slen = int(batch["slen"].unique().item())
        true_params = {k: v for k, v in batch.items() if k not in exclude}

        # get params on full image.
        true_params = get_full_params(true_params, slen)

        # get map estimates
        tile_estimate = self.tile_map_estimate(batch)
        estimates = get_full_params(tile_estimate, slen)
        recon_images, _ = self.image_decoder.render_images(
            tile_estimate["n_sources"],
            tile_estimate["locs"],
            tile_estimate["galaxy_bool"],
            tile_estimate["galaxy_params"],
            tile_estimate["fluxes"],
            add_noise=False,
        )

        # get detection and star fluxes metrics
        errors = eval_error_on_batch(true_params, estimates, slen)

        locs_mae = errors["locs_mae_vec"].mean()
        star_fluxes_mae = errors["fluxes_mae_vec"].mean()
        counts_acc = errors["count_bool"].float().mean()
        galaxy_counts_acc = errors["galaxy_counts_bool"].float().mean()
        avg_tpr = errors["tpr_vec"].mean()
        avg_ppv = errors["ppv_vec"].mean()

        # galaxy metrics
        gal_params_mae = 0.0
        if self.use_galaxy_encoder:
            gal_params_mae = errors["galaxy_params_mae_vec"].float().mean()

        # image metrics
        background = self.image_decoder.background
        image_diff = true_images - recon_images
        diff_fluxes = true_images.sum((1, 2, 3)) - recon_images.sum((1, 2, 3))
        # TODO: normalize the expression below?
        image_fluxes_mae = diff_fluxes.abs().mean()
        norm_pp_mae = (
            (image_diff.sum((1, 2, 3)) / (true_images - background).sum((1, 2, 3)))
            .abs()
            .mean()
        )

        return {
            "counts_acc": counts_acc,
            "galaxy_counts_acc": galaxy_counts_acc,
            "locs_mae": locs_mae,
            "star_fluxes_mae": star_fluxes_mae,
            "avg_tpr": avg_tpr,
            "avg_ppv": avg_ppv,
            "galaxy_params_mae": gal_params_mae,
            "image_fluxes_mae": image_fluxes_mae,
            "norm_pp_mae": norm_pp_mae,
        }

    # pylint: disable=too-many-statements
    def make_plots(self, batch, kind="validation"):
        # add some images to tensorboard for validating location/counts.
        # 'batch' is a batch from simulated dataset (all params are tiled)
        n_samples = min(10, len(batch["n_sources"]))
        assert n_samples > 1

        # extract non-params entries for get full params to works.
        exclude = {"images", "slen", "background"}
        images = batch["images"]
        slen = int(batch["slen"].unique().item())
        border_padding = int((images.shape[-1] - slen) / 2)
        _true_params = {k: v for k, v in batch.items() if k not in exclude}
        true_params = get_full_params(_true_params, slen)

        # obtain map estimates
        tile_estimate = self.tile_map_estimate(batch)
        estimate = get_full_params(tile_estimate, slen)
        assert len(estimate["locs"].shape) == 3
        assert estimate["locs"].shape[1] == estimate["n_sources"].max().int().item()

        figsize = (12, 4 * n_samples)
        fig, axes = plt.subplots(nrows=n_samples, ncols=3, figsize=figsize)

        for i in range(n_samples):
            true_ax = axes[i, 0]
            recon_ax = axes[i, 1]
            res_ax = axes[i, 2]

            image = images[None, i]
            assert len(image.shape) == 4

            # true parameters on full image.
            true_n_sources = true_params["n_sources"][None, i]
            true_locs = true_params["locs"][None, i]
            true_galaxy_bool = true_params["galaxy_bool"][None, i]

            # convert tile estimates to full parameterization for plotting
            n_sources = estimate["n_sources"][None, i]
            locs = estimate["locs"][None, i]
            galaxy_bool = estimate["galaxy_bool"][None, i]

            # plot true image + number of sources first.
            image = image[0, 0].cpu().numpy()  # only first band will be plotted.
            plotting.plot_image(fig, true_ax, image)
            true_ax.set_xlabel(
                f"True num: {true_n_sources.item()}; Est num: {n_sources.item()}"
            )

            # continue only if at least one true source and predicted source.
            max_sources = true_locs.shape[1]
            if max_sources > 0 and n_sources.item() > 0:
                # draw reconstruction image.
                recon_image, _ = self.image_decoder.render_images(
                    tile_estimate["n_sources"][None, i],
                    tile_estimate["locs"][None, i],
                    tile_estimate["galaxy_bool"][None, i],
                    tile_estimate["galaxy_params"][None, i],
                    tile_estimate["fluxes"][None, i],
                    add_noise=False,
                )

                # round up true parameters.
                true_star_bool = get_star_bool(true_n_sources, true_galaxy_bool)
                true_galaxy_locs = true_locs * true_galaxy_bool
                true_star_locs = true_locs * true_star_bool

                # round up estimated parameters.
                star_bool = get_star_bool(n_sources, galaxy_bool)
                galaxy_locs = locs * galaxy_bool
                star_locs = locs * star_bool

                # convert everything to numpy + cpu so matplotlib can use it.
                true_galaxy_locs = true_galaxy_locs.cpu().numpy()[0]
                true_star_locs = true_star_locs.cpu().numpy()[0]
                galaxy_locs = galaxy_locs.cpu().numpy()[0]
                star_locs = star_locs.cpu().numpy()[0]

                recon_image = recon_image[0, 0].cpu().numpy()
                res_image = (image - recon_image) / np.sqrt(image)

                # plot and add locations.
                plotting.plot_image_locs(
                    true_ax,
                    slen,
                    border_padding,
                    true_locs=true_galaxy_locs,
                    est_locs=galaxy_locs,
                    markers=("x", "+"),
                )
                plotting.plot_image_locs(
                    true_ax,
                    slen,
                    border_padding,
                    true_locs=true_star_locs,
                    est_locs=star_locs,
                    markers=("o", "1"),
                )

                plotting.plot_image(fig, recon_ax, recon_image)
                plotting.plot_image(fig, res_ax, res_image)

            else:
                zero_image = np.zeros((images.shape[-1], images.shape[-1]))
                plotting.plot_image(fig, recon_ax, zero_image)
                plotting.plot_image(fig, res_ax, zero_image)

        plt.subplots_adjust(hspace=0.2, wspace=0.4)
        if self.logger:
            if kind == "validation":
                title = f"Val Images {self.current_epoch}"
                self.logger.experiment.add_figure(title, fig)
            elif kind == "testing":
                self.logger.experiment.add_figure("Test Images", fig)
            else:
                raise NotImplementedError()
        plt.close(fig)
