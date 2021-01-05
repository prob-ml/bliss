import math
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pytorch_lightning as pl

import torch
from torch.nn import CrossEntropyLoss
from torch.distributions import Normal
from torch.optim import Adam

from . import device, plotting
from .models import encoder, decoder
from .models.encoder import get_star_bool

plt.switch_backend("Agg")


def sort_locs(locs):
    # sort according to x location
    assert len(locs.shape) == 2
    indx_sort = locs[:, 0].sort()[1]
    return locs[indx_sort, :], indx_sort


def _get_log_probs_all_perms(
    locs_log_probs_all,
    galaxy_params_log_probs_all,
    star_params_log_probs_all,
    prob_galaxy,
    true_galaxy_bool,
    is_on_array,
):
    # get log-probability under every possible matching of estimated source to true source
    n_ptiles = galaxy_params_log_probs_all.size(0)
    max_detections = galaxy_params_log_probs_all.size(-1)

    n_permutations = math.factorial(max_detections)
    locs_log_probs_all_perm = torch.zeros(n_ptiles, n_permutations, device=device)
    galaxy_params_log_probs_all_perm = locs_log_probs_all_perm.clone()
    star_params_log_probs_all_perm = locs_log_probs_all_perm.clone()
    galaxy_bool_log_probs_all_perm = locs_log_probs_all_perm.clone()

    for i, perm in enumerate(permutations(range(max_detections))):
        # note that we multiply is_on_array, we only evaluate the loss if the source is on.
        locs_log_probs_all_perm[:, i] = (
            locs_log_probs_all[:, perm].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)

        # if galaxy, evaluate the galaxy parameters,
        # hence the multiplication by (true_galaxy_bool)
        # the diagonal is a clever way of selecting the elements of each permutation (first index
        # of mean/var with second index of true_param etc.)
        galaxy_params_log_probs_all_perm[:, i] = (
            galaxy_params_log_probs_all[:, perm].diagonal(dim1=1, dim2=2)
            * is_on_array
            * true_galaxy_bool
        ).sum(1)

        # similarly for stars
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
        galaxy_params_log_probs_all_perm,
        star_params_log_probs_all_perm,
        galaxy_bool_log_probs_all_perm,
    )


def _get_min_perm_loss(
    locs_log_probs_all,
    galaxy_params_log_probs_all,
    star_params_log_probs_all,
    prob_galaxy,
    true_galaxy_bool,
    is_on_array,
):
    # get log-probability under every possible matching of estimated star to true star
    (
        locs_log_probs_all_perm,
        galaxy_params_log_probs_all_perm,
        star_params_log_probs_all_perm,
        galaxy_bool_log_probs_all_perm,
    ) = _get_log_probs_all_perms(
        locs_log_probs_all,
        galaxy_params_log_probs_all,
        star_params_log_probs_all,
        prob_galaxy,
        true_galaxy_bool,
        is_on_array,
    )

    # TODO: Why do we select it based on the location losses only?
    # find the permutation that minimizes the location losses
    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim=1)

    # get the star & galaxy losses according to the found permutation
    _indx = indx.unsqueeze(1)
    star_params_loss = -torch.gather(star_params_log_probs_all_perm, 1, _indx).squeeze()
    galaxy_params_loss = -torch.gather(
        galaxy_params_log_probs_all_perm, 1, _indx
    ).squeeze()
    galaxy_bool_loss = -torch.gather(galaxy_bool_log_probs_all_perm, 1, _indx).squeeze()

    return locs_loss, galaxy_params_loss, star_params_loss, galaxy_bool_loss


class SleepPhase(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(SleepPhase, self).__init__()
        self.save_hyperparameters(cfg)

        self.image_encoder = encoder.ImageEncoder(**cfg.model.encoder.params)
        self.image_decoder = decoder.ImageDecoder(**cfg.model.decoder.params)
        self.image_decoder.requires_grad_(False)

        self.plotting: bool = cfg.training.plotting

        # consistency
        assert self.image_decoder.n_galaxy_params == self.image_encoder.n_galaxy_params
        assert self.image_decoder.tile_slen == self.image_encoder.tile_slen
        assert self.image_decoder.border_padding == self.image_encoder.border_padding

    def forward(self, image_ptiles, n_sources):
        raise NotImplementedError()

    def map_estimate(self, slen, images, batch):
        # batch is per tile since it comes from image_decoder
        tile_galaxy_params = {}
        tile_params = {
            "galaxy_bool": batch["galaxy_bool"],
            "n_sources": batch["n_sources"],
        }
        if self.image_encoder:
            tile_n_sources = self.image_encoder.tile_map_n_sources(images)
            tile_params = self.image_encoder.tile_map_estimate(images, tile_n_sources)
            tile_params["n_sources"] = tile_n_sources
            tile_params["galaxy_bool"] = tile_params["galaxy_bool"]

        if self.galaxy_encoder:
            tile_galaxy_params = self.galaxy_encoder.tile_map_estimate(
                images, tile_params["n_sources"], tile_params["galaxy_bool"]
            )

        tile_est = {**tile_params, **tile_galaxy_params}
        est = get_full_params(slen, tile_est)
        return est

    def get_loss(self, batch):
        """

        loc_mean shape = (n_ptiles x max_detections x 2)
        log_flux_mean shape = (n_ptiles x max_detections x n_bands)
        galaxy_param_mean shape = (n_ptiles x max_detections x n_galaxy_params)

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
            true_tile_galaxy_params,
            true_tile_log_fluxes,
            true_tile_galaxy_bool,
            true_tile_n_sources,
        ) = (
            batch["images"],
            batch["locs"],
            batch["galaxy_params"],
            batch["log_fluxes"],
            batch["galaxy_bool"],
            batch["n_sources"],
        )

        # some constants
        batch_size = images.shape[0]
        n_tiles_per_image = self.image_decoder.n_tiles_per_image
        n_ptiles = batch_size * n_tiles_per_image
        max_sources_dec = self.image_decoder.max_sources
        max_sources = self.image_encoder.max_detections
        n_bands = self.image_decoder.n_bands
        n_galaxy_params = self.image_decoder.n_galaxy_params

        # clip to max sources
        if max_sources < max_sources_dec:
            true_tile_locs = true_tile_locs[:, :, 0:max_sources]
            true_tile_galaxy_params = true_tile_galaxy_params[:, :, 0:max_sources]
            true_tile_log_fluxes = true_tile_log_fluxes[:, :, 0:max_sources]
            true_tile_galaxy_bool = true_tile_galaxy_bool[:, :, 0:max_sources]
            true_tile_n_sources = true_tile_n_sources.clamp(max=max_sources)

        # flatten so first dimension is ptile
        true_tile_locs = true_tile_locs.view(n_ptiles, max_sources, 2)
        true_tile_galaxy_params = true_tile_galaxy_params.view(
            n_ptiles, max_sources, n_galaxy_params
        )
        true_tile_log_fluxes = true_tile_log_fluxes.view(n_ptiles, max_sources, n_bands)
        true_tile_galaxy_bool = true_tile_galaxy_bool.view(n_ptiles, max_sources)
        true_tile_n_sources = true_tile_n_sources.view(n_ptiles)
        true_tile_is_on_array = encoder.get_is_on_from_n_sources(
            true_tile_n_sources, max_sources
        )

        # extract image tiles
        # true_tile_locs has shape = (n_ptiles x max_detections x 2)
        # true_tile_n_sources has shape = (n_ptiles)
        image_ptiles = self.image_encoder.get_images_in_tiles(images)
        pred = self(image_ptiles, true_tile_n_sources)

        # the loss for estimating the true number of sources
        n_source_log_probs = pred["n_source_log_probs"].view(n_ptiles, max_sources + 1)
        cross_entropy = CrossEntropyLoss(reduction="none").requires_grad_(False)
        counter_loss = cross_entropy(n_source_log_probs, true_tile_n_sources)

        # the following three functions computes the log-probability of parameters when
        # each estimated source i is matched with true source j for
        # i, j in {1, ..., max_detections}
        # *_log_probs_all have shape n_ptiles x max_detections x max_detections

        # enforce large error if source is off
        loc_mean, loc_logvar = pred["loc_mean"], pred["loc_logvar"]
        loc_mean = loc_mean + (true_tile_is_on_array == 0).float().unsqueeze(-1) * 1e16
        locs_log_probs_all = self._get_params_logprob_all_combs(
            true_tile_locs, loc_mean, loc_logvar
        )

        star_params_log_probs_all = self._get_params_logprob_all_combs(
            true_tile_log_fluxes, pred["log_flux_mean"], pred["log_flux_logvar"]
        )

        # inside _get_min_perm_loss is where the matching happens:
        # we construct a bijective map from each estimated source to each true source
        prob_galaxy = pred["prob_galaxy"].view(n_ptiles, max_sources)
        (
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = _get_min_perm_loss(
            locs_log_probs_all,
            galaxy_params_log_probs_all,
            star_params_log_probs_all,
            prob_galaxy,
            true_tile_galaxy_bool,
            true_tile_is_on_array,
        )

        # calculate kl_qp loss.
        # TODO: Should I zero out with is_on_array?
        q_z = Normal(
            pred["galaxy_param_mean"], pred["galaxy_param_logvar"].exp().sqrt()
        )
        z = q_z.rsample()
        log_q_z = q_z.log_prob(z).sum((1, 2))
        p_z = Normal(torch.zeros(z.shape), torch.ones(z.shape))
        log_p_z = p_z.log_prob(z).sum((1, 2))

        # TODO: Are true_tile_locs matched with z? Does this matter or do we have to do
        #  permutation stuff?
        recon_mean, recon_var = self.image_decoder.render_images(
            true_tile_n_sources,
            true_tile_locs,
            true_tile_galaxy_bool,
            z,
            true_tile_log_fluxes.exp(),
        )

        kl_z = log_q_z - log_p_z  # log q(z | x) - log p(z)
        recon_losses = -Normal(recon_mean, recon_var.sqrt()).log_prob(images)
        recon_losses = recon_losses.view(n_ptiles, -1).sum(1)

        kl_qp = (recon_losses + kl_z).sum()

        loss_vec = (
            locs_loss * (locs_loss.detach() < 1e6).float()
            + counter_loss
            + kl_qp
            + star_params_loss
            + galaxy_bool_loss
        )

        loss = loss_vec.mean()

        return (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        )

    def configure_optimizers(self):
        # TODO: use different optimizer parameters for each encoder.
        params = self.hparams.optimizer.params
        image_opt = Adam(self.image_encoder.parameters(), **params)
        galaxy_opt = Adam(self.galaxy_encoder.parameters(), **params)
        return image_opt, galaxy_opt

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:  # image_encoder
            (
                loss,
                counter_loss,
                locs_loss,
                galaxy_params_loss,
                star_params_loss,
                galaxy_bool_loss,
            ) = self.get_loss(batch)
            self.log("train_loss", loss)
            return loss

        if optimizer_idx == 1:  # galaxy_encoder:
            loss = 0.0
            return loss

    def validation_step(self, batch, batch_indx):
        (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = self.get_loss(batch)

        self.log("val_loss", loss)
        self.log("val_counter_loss", counter_loss.mean())
        self.log("val_gal_bool_loss", galaxy_bool_loss.mean())
        self.log("val_star_params_loss", star_params_loss.mean())
        self.log("val_gal_params_loss", galaxy_params_loss.mean())

        # calculate metrics for this batch
        (
            counts_acc,
            galaxy_counts_acc,
            locs_median_mse,
            fluxes_avg_err,
        ) = self.get_metrics(batch)
        self.log("val_acc_counts", counts_acc)
        self.log("val_gal_counts", galaxy_counts_acc)
        self.log("val_locs_median_mse", locs_median_mse)
        self.log("val_fluxes_avg_err", fluxes_avg_err)
        return batch

    def validation_epoch_end(self, outputs):
        # NOTE: outputs is a list containing all validation step batches.
        if self.plotting and self.current_epoch > 1:
            self.make_plots(outputs[-1], kind="validation")

    def test_step(self, batch, batch_indx):
        (
            counts_acc,
            galaxy_counts_acc,
            locs_median_mse,
            fluxes_avg_err,
        ) = self.get_metrics(batch)
        self.log("acc_counts", counts_acc)
        self.log("acc_gal_counts", galaxy_counts_acc)
        self.log("locs_median_mse", locs_median_mse)
        self.log("fluxes_avg_err", fluxes_avg_err)

        return batch

    def test_epoch_end(self, outputs):
        batch = outputs[-1]
        if self.plotting:
            self.make_plots(batch, kind="testing")

    def get_metrics(self, batch):
        # get images and properties
        exclude = {"images", "slen", "background"}
        images = batch["images"]
        slen = int(batch["slen"].unique().item())
        n_bands = batch["images"].shape[1]
        batch_size = images.shape[0]
        true_params = {k: v for k, v in batch.items() if k not in exclude}

        # get params on full image.
        true_params = get_full_params(slen, true_params)

        # get map estimates
        estimates = self.image_encoder.map_estimate(slen, images)

        # accuracy of counts
        counts_acc = true_params["n_sources"].eq(estimates["n_sources"]).float().mean()

        # accuracy of galaxy counts
        est_n_gal = estimates["galaxy_bool"].view(batch_size, -1).sum(-1)
        true_n_gal = estimates["galaxy_bool"].view(batch_size, -1).sum(-1)
        galaxy_counts_acc = est_n_gal.eq(true_n_gal).float().mean()

        # accuracy of locations
        est_locs = estimates["locs"]
        true_locs = true_params["locs"]

        # accuracy of fluxes
        est_fluxes = estimates["fluxes"]
        true_fluxes = true_params["fluxes"]

        locs_mse_vec = []
        fluxes_mse_vec = []
        for i in range(batch_size):
            true_n_sources_i = true_params["n_sources"][i]
            n_sources_i = estimates["n_sources"][i]

            # only compare locations for mse if counts match.
            if true_n_sources_i == n_sources_i:

                # prepare locs and get them in units of pixels.
                true_locs_i = true_locs[i].view(-1, 2)
                true_locs_i = true_locs_i[: int(true_n_sources_i)] * slen
                locs_i = est_locs[i].view(-1, 2)[: int(n_sources_i)] * slen

                # sort each based on x location.
                true_locs_i, indx_sort_true = sort_locs(true_locs_i)
                locs_i, indx_sort = sort_locs(locs_i)

                # now calculate mse
                locs_mse = (true_locs_i - locs_i).pow(2).sum(1).pow(1.0 / 2)
                for mse in locs_mse:
                    locs_mse_vec.append(mse.item())

                # do the same for fluxes
                true_fluxes_i = true_fluxes[i].view(-1, n_bands)
                true_fluxes_i = true_fluxes_i[: int(true_n_sources_i)]
                fluxes_i = est_fluxes[i].view(-1, n_bands)[: int(n_sources_i)]

                # sort the same way we did locations
                true_fluxes_i = true_fluxes_i[indx_sort_true]
                fluxes_i = fluxes_i[indx_sort_true]

                # convert to magnitude and compute error
                true_mags_i = torch.log10(true_fluxes_i) * 2.5
                log_mags_i = torch.log10(fluxes_i) * 2.5
                fluxes_mse = torch.abs(true_mags_i - log_mags_i).mean(1)
                for mse in fluxes_mse:
                    fluxes_mse_vec.append(mse.item())

        # TODO: default value? Also not sure how to accumulate medians so we are actually taking an
        #  average over the medians across batches.
        locs_median_mse = 0.5
        if len(locs_mse_vec) > 0:
            locs_median_mse = np.median(locs_mse_vec)

        fluxes_avg_err = 1e16
        if len(fluxes_mse_vec) > 0:
            fluxes_avg_err = np.mean(fluxes_mse_vec)

        return counts_acc, galaxy_counts_acc, locs_median_mse, fluxes_avg_err

    def make_plots(self, batch, kind="validation"):
        # add some images to tensorboard for validating location/counts.
        # 'batch' is a batch from simulated dataset (all params are tiled)
        n_samples = min(10, len(batch["n_sources"]))
        assert n_samples > 1

        # extract non-params entries get_full_params works.
        exclude = {"images", "slen", "background"}
        images = batch["images"]
        slen = int(batch["slen"].unique().item())
        border_padding = int((images.shape[-1] - slen) / 2)
        true_params = {k: v for k, v in batch.items() if k not in exclude}

        # convert to full image parameters for plotting purposes.
        true_params = get_full_params(slen, true_params)

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

            with torch.no_grad():
                # get the estimated params, these are *per tile*.
                self.image_encoder.eval()
                tile_estimate = self.image_encoder.tiled_map_estimate(image)

            # convert tile estimates to full parameterization for plotting
            estimate = get_full_params(slen, tile_estimate)
            n_sources = estimate["n_sources"]
            locs = estimate["locs"]
            galaxy_bool = estimate["galaxy_bool"]

            assert len(locs.shape) == 3 and locs.size(0) == 1
            assert locs.shape[1] == n_sources.max().int().item()

            # plot true image + number of sources first.
            image = image[0, 0].cpu().numpy()  # only first band.
            plotting.plot_image(fig, true_ax, image)
            true_ax.set_xlabel(
                f"True num: {true_n_sources.item()}; Est num: {n_sources.item()}"
            )

            # continue only if at least one true source and predicted source.
            max_sources = true_locs.shape[1]
            if max_sources > 0 and n_sources.item() > 0:

                # draw reconstruction image.
                recon_image, _ = self.image_decoder.render_images(
                    tile_estimate["n_sources"],
                    tile_estimate["locs"],
                    tile_estimate["galaxy_bool"],
                    tile_estimate["galaxy_params"],
                    tile_estimate["fluxes"],
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
                    colors=("r", "b"),
                )
                plotting.plot_image_locs(
                    true_ax,
                    slen,
                    border_padding,
                    true_locs=true_star_locs,
                    est_locs=star_locs,
                    colors=("lightpink", "tab:orange"),
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
                self.logger.experiment.add_figure(f"Test Images", fig)
            else:
                raise NotImplementedError()
        plt.close(fig)

    @staticmethod
    def _get_params_logprob_all_combs(true_params, param_mean, param_logvar):
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
