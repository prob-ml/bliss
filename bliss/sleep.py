"""Summary for sleep.py

This module contains SleepPhase class, which implements the sleep-phase traning using
pytorch-lightning framework. Users should use this class to construct the sleep-phase
model.

"""

import math
from itertools import permutations

import matplotlib.pyplot as plt
import pytorch_lightning as pl

import torch
from torch.nn import CrossEntropyLoss
from torch.distributions import Normal
from einops import rearrange

from bliss import plotting
from bliss.optimizer import get_optimizer
from bliss.models import encoder, decoder
from bliss.models.encoder import get_star_bool, get_full_params
from bliss.metrics import eval_error_on_batch

plt.switch_backend("Agg")


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

    def __init__(
        self,
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        annotate_probs: bool = False,
        optimizer_params: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = encoder.ImageEncoder(**encoder_kwargs)
        self.image_decoder = decoder.ImageDecoder(**decoder_kwargs)
        self.image_decoder.requires_grad_(False)
        self.optimizer_params = optimizer_params

        # consistency
        assert self.image_decoder.tile_slen == self.image_encoder.tile_slen
        assert self.image_decoder.border_padding == self.image_encoder.border_padding
        assert self.image_encoder.max_detections <= self.image_decoder.max_sources

        # plotting
        self.annotate_probs = annotate_probs

    def forward(self, image_ptiles, tile_n_sources):
        return self.image_encoder(image_ptiles, tile_n_sources)

    def tile_map_estimate(self, batch):
        images = batch["images"]
        tile_est = self.image_encoder.tile_map_estimate(images)
        tile_est["galaxy_params"] = batch["galaxy_params"]

        # FIXME: True galaxy params are not necessarily consistent with MAP estimates
        # need to do some matching to ensure correctness of residual images?
        # maybe doesn't matter because only care about detection if not estimating
        # galaxy_parameters.
        max_sources = tile_est["locs"].shape[2]
        tile_est["galaxy_params"] = tile_est["galaxy_params"][:, :, :max_sources]
        tile_est["galaxy_params"] = tile_est["galaxy_params"].contiguous()
        return tile_est

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

        # clip decoder output since constraint is: max_detections <= max_sources (per tile)
        true_tile_locs = true_tile_locs[:, :, 0:max_sources]
        true_tile_log_fluxes = true_tile_log_fluxes[:, :, 0:max_sources]
        true_tile_galaxy_bool = true_tile_galaxy_bool[:, :, 0:max_sources]
        true_tile_n_sources = true_tile_n_sources.clamp(max=max_sources)

        # flatten so first dimension is ptile
        # b: batch, s: n_tiles_per_image
        true_tile_locs = rearrange(true_tile_locs, "b n s xy -> (b n) s xy", xy=2)
        true_tile_log_fluxes = rearrange(true_tile_log_fluxes, "b n s bands -> (b n) s bands")
        true_tile_galaxy_bool = rearrange(true_tile_galaxy_bool, "b n s 1 -> (b n) s")
        true_tile_n_sources = rearrange(true_tile_n_sources, "b n -> (b n)")
        true_tile_is_on_array = encoder.get_is_on_from_n_sources(true_tile_n_sources, max_sources)

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
        locs_log_probs_all = _get_params_logprob_all_combs(true_tile_locs, loc_mean, loc_logvar)
        star_params_log_probs_all = _get_params_logprob_all_combs(
            true_tile_log_fluxes, pred["log_flux_mean"], pred["log_flux_logvar"]
        )
        prob_galaxy = rearrange(pred["prob_galaxy"], "bn s 1 -> bn s")

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
        assert self.optimizer_params is not None, "Need to specify `optimizer_params`."
        name = self.optimizer_params["name"]
        kwargs = self.optimizer_params["kwargs"]
        opt = get_optimizer(name, self.image_encoder.parameters(), kwargs)
        return opt

    def training_step(self, batch, batch_idx, optimizer_idx=0):  # pylint: disable=unused-argument
        loss = self.get_detection_loss(batch)[0]
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        (
            detection_loss,
            counter_loss,
            locs_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = self.get_detection_loss(batch)

        self.log("val/loss", detection_loss)
        self.log("val/counter_loss", counter_loss.mean())
        self.log("val/locs_loss", locs_loss.mean())
        self.log("val/gal_bool_loss", galaxy_bool_loss.mean())
        self.log("val/star_params_loss", star_params_loss.mean())

        # calculate metrics for this batch
        metrics = self.get_metrics(batch)
        self.log("val/acc_counts", metrics["counts_acc"])
        self.log("val/gal_counts", metrics["galaxy_counts_acc"])
        self.log("val/locs_mae", metrics["locs_mae"])
        self.log("val/star_fluxes_mae", metrics["star_fluxes_mae"])
        self.log("val/avg_tpr", metrics["avg_tpr"])
        self.log("val/avg_ppv", metrics["avg_ppv"])
        return batch

    def validation_epoch_end(self, outputs):
        if self.current_epoch > 1:
            self.make_plots(outputs[-1], kind="validation")

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        metrics = self.get_metrics(batch)
        self.log("acc_counts", metrics["counts_acc"])
        self.log("acc_gal_counts", metrics["galaxy_counts_acc"])
        self.log("locs_mae", metrics["locs_mae"])
        self.log("star_fluxes_mae", metrics["star_fluxes_mae"])
        self.log("avg_tpr", metrics["avg_tpr"])
        self.log("avg_ppv", metrics["avg_ppv"])

        return batch

    def test_epoch_end(self, outputs):
        batch = outputs[-1]
        self.make_plots(batch, kind="testing")

    def get_metrics(self, batch):
        # get images and properties
        exclude = {"images", "slen", "background"}
        slen = int(batch["slen"].unique().item())
        true_params = {k: v for k, v in batch.items() if k not in exclude}

        # get params on full image.
        true_params = get_full_params(true_params, slen)

        # get map estimates
        tile_estimate = self.tile_map_estimate(batch)
        estimates = get_full_params(tile_estimate, slen)

        # get detection and star fluxes metrics
        errors = eval_error_on_batch(true_params, estimates, slen)

        locs_mae = errors["locs_mae_vec"].mean()
        star_fluxes_mae = errors["fluxes_mae_vec"].mean()
        counts_acc = errors["count_bool"].float().mean()
        galaxy_counts_acc = errors["galaxy_counts_bool"].float().mean()
        avg_tpr = errors["tpr_vec"].mean()
        avg_ppv = errors["ppv_vec"].mean()

        return {
            "counts_acc": counts_acc,
            "galaxy_counts_acc": galaxy_counts_acc,
            "locs_mae": locs_mae,
            "star_fluxes_mae": star_fluxes_mae,
            "avg_tpr": avg_tpr,
            "avg_ppv": avg_ppv,
        }

    # pylint: disable=too-many-statements
    def make_plots(self, batch, kind="validation"):
        # add some images to tensorboard for validating location/counts.
        # 'batch' is a batch from simulated dataset (all params are tiled)
        n_samples = 10

        if n_samples > len(batch["n_sources"]):  # do nothing if low on samples.
            return

        # extract non-params entries so that 'get_full_params' to works.
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
        figsize = (20, 8)
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=figsize)
        axes = axes.flatten()

        for i in range(n_samples):

            ax = axes[i]

            image = images[i, 0].cpu().numpy()

            # true parameters on full image.
            true_n_sources = true_params["n_sources"][None, i]
            true_locs = true_params["locs"][None, i]
            true_galaxy_bool = true_params["galaxy_bool"][None, i]

            # convert tile estimates to full parameterization for plotting
            n_sources = estimate["n_sources"][None, i]
            locs = estimate["locs"][None, i]
            galaxy_bool = estimate["galaxy_bool"][None, i]
            prob_galaxy = estimate["prob_galaxy"][None, i]

            # round up true and estimated parameters.
            true_star_bool = get_star_bool(true_n_sources, true_galaxy_bool)
            true_galaxy_locs = true_locs * true_galaxy_bool
            true_star_locs = true_locs * true_star_bool

            star_bool = get_star_bool(n_sources, galaxy_bool)
            galaxy_locs = locs * galaxy_bool
            star_locs = locs * star_bool

            # convert everything to numpy + cpu so matplotlib can use it.
            true_galaxy_locs = true_galaxy_locs.cpu().numpy()[0]
            true_star_locs = true_star_locs.cpu().numpy()[0]
            galaxy_locs = galaxy_locs.cpu().numpy()[0]
            star_locs = star_locs.cpu().numpy()[0]
            prob_galaxy = prob_galaxy.cpu().numpy()[0].reshape(-1)

            # x-axis comparing true objects and estimated objects.
            ax.set_xlabel(f"True num: {true_n_sources.item()}; Est num: {n_sources.item()}")

            # plot these images too.
            plotting.plot_image(fig, ax, image, vmin=image.min().item(), vmax=image.max().item())

            # galaxies first
            plotting.plot_image_locs(
                ax,
                slen,
                border_padding,
                true_locs=true_galaxy_locs,
                est_locs=galaxy_locs,
                prob_galaxy=prob_galaxy if self.annotate_probs else None,
                markers=("x", "+"),
                colors=("r", "b"),
            )

            # then stars
            plotting.plot_image_locs(
                ax,
                slen,
                border_padding,
                true_locs=true_star_locs,
                est_locs=star_locs,
                prob_galaxy=prob_galaxy if self.annotate_probs else None,
                markers=("x", "+"),
                colors=("orange", "deepskyblue"),
            )

        fig.tight_layout()
        if self.logger:
            if kind == "validation":
                title = f"Val Images {self.current_epoch}"
                self.logger.experiment.add_figure(title, fig)
            elif kind == "testing":
                self.logger.experiment.add_figure("(Worst) Test Images", fig)
            else:
                raise NotImplementedError()
        plt.close(fig)
