import os
import math
import numpy as np
from itertools import permutations
import inspect
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal
from torch.nn import functional
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Callback

from . import device, plotting
from .models import encoder

import optuna
from optuna.integration import PyTorchLightningPruningCallback


DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def _get_categorical_loss(n_source_log_probs, one_hot_encoding):
    assert torch.all(n_source_log_probs <= 0)
    assert n_source_log_probs.shape == one_hot_encoding.shape

    return -torch.sum(n_source_log_probs * one_hot_encoding, dim=1)


def _get_params_logprob_all_combs(true_params, param_mean, param_logvar):
    assert true_params.shape == param_mean.shape == param_logvar.shape

    n_ptiles = true_params.size(0)
    max_detections = true_params.size(1)

    # reshape to evaluate all combinations of log_prob.
    _true_params = true_params.view(n_ptiles, 1, max_detections, -1)
    _param_mean = param_mean.view(n_ptiles, max_detections, 1, -1)
    _param_logvar = param_logvar.view(n_ptiles, max_detections, 1, -1)

    _sd = (_param_logvar.exp() + 1e-5).sqrt()
    param_log_probs_all = Normal(_param_mean, _sd).log_prob(_true_params).sum(dim=3)
    return param_log_probs_all


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
    def __init__(
        self,
        dataset,
        encoder_kwargs,
        lr=1e-3,
        weight_decay=1e-5,
        validation_plot_start=5,
    ):
        super(SleepPhase, self).__init__()

        # assumes dataset is a IterableDataset class.
        self.dataset = dataset
        self.image_encoder = encoder.ImageEncoder(**encoder_kwargs)

        # avoid calculating gradients of psf_transform
        self.dataset.image_decoder.power_law_psf.requires_grad_(False)

        self.lr = lr
        self.weight_decay = weight_decay

        self.validation_plot_start = validation_plot_start
        assert self.dataset.latent_dim == self.image_encoder.n_galaxy_params

        self.hparams = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.dataset.batch_size,
            "n_batches": self.dataset.n_batches,
            "n_bands": self.dataset.n_bands,
            "max_sources": self.dataset.image_decoder.max_sources,
            "mean_sources": self.dataset.image_decoder.mean_sources,
            "min_sources": self.dataset.image_decoder.min_sources,
            "loc_min": self.dataset.image_decoder.loc_min,
            "loc_max": self.dataset.image_decoder.loc_max,
            "prob_galaxy": self.dataset.image_decoder.prob_galaxy,
        }

    def forward(self, image_ptiles, n_sources):
        return self.image_encoder.forward(image_ptiles, n_sources)

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

        true_galaxy_bool shape = (n_ptiles x max_detections)
            indicating whether each source is a galaxy (1) or star (0)

        prob_galaxy shape = (n_ptiles x max_detections)
            are probabilities for each source to be a galaxy

        n_source_log_probs shape = (n_ptiles x (max_detections + 1))
            are log-probabilities for the number of sources (0, 1, ..., max_detections)

        """
        (images, true_locs, true_galaxy_params, true_log_fluxes, true_galaxy_bool,) = (
            batch["images"],
            batch["locs"],
            batch["galaxy_params"],
            batch["log_fluxes"],
            batch["galaxy_bool"],
        )

        # extract image tiles
        # true_tile_locs has shape = (n_ptiles x max_detections x 2)
        # true_tile_n_sources has shape = (n_ptiles)
        slen = images.size(-1)
        image_ptiles = self.image_encoder.get_images_in_tiles(images)
        (
            true_tile_n_sources,
            true_tile_locs,
            true_tile_galaxy_params,
            true_tile_log_fluxes,
            true_tile_galaxy_bool,
            true_tile_is_on_array,
        ) = self.image_encoder.get_params_in_tiles(
            slen, true_locs, true_galaxy_params, true_log_fluxes, true_galaxy_bool
        )

        n_ptiles = true_tile_is_on_array.size(0)
        max_detections = true_tile_is_on_array.size(1)

        pred = self.forward(image_ptiles, true_tile_n_sources)

        # TODO: make .forward() and .get_params_in_tiles() just return correct dimensions ?
        prob_galaxy = pred["prob_galaxy"].view(n_ptiles, max_detections)
        true_tile_galaxy_bool = true_tile_galaxy_bool.view(n_ptiles, max_detections)

        # the loss for estimating the true number of sources
        n_source_log_probs = pred["n_source_log_probs"]
        true_tile_n_sources = true_tile_is_on_array.sum(1).long()  # per tile.
        one_hot_encoding = functional.one_hot(true_tile_n_sources, max_detections + 1)
        counter_loss = _get_categorical_loss(n_source_log_probs, one_hot_encoding)

        # the following three functions computes the log-probability of parameters when
        # each estimated source i is matched with true source j for
        # i, j in {1, ..., max_detections}
        # *_log_probs_all have shape n_ptiles x max_detections x max_detections

        # enforce large error if source is off
        loc_mean, loc_logvar = pred["loc_mean"], pred["loc_logvar"]
        loc_mean = loc_mean + (true_tile_is_on_array == 0).float().unsqueeze(-1) * 1e16
        locs_log_probs_all = _get_params_logprob_all_combs(
            true_tile_locs, loc_mean, loc_logvar
        )
        galaxy_params_log_probs_all = _get_params_logprob_all_combs(
            true_tile_galaxy_params,
            pred["galaxy_param_mean"],
            pred["galaxy_param_logvar"],
        )
        star_params_log_probs_all = _get_params_logprob_all_combs(
            true_tile_log_fluxes, pred["log_flux_mean"], pred["log_flux_logvar"]
        )

        # inside _get_min_perm_loss is where the matching happens:
        # we construct a bijective map from each estimated source to each true source
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

        loss_vec = (
            locs_loss * (locs_loss.detach() < 1e6).float()
            + counter_loss
            # + galaxy_params_loss
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
        return Adam(
            [{"params": self.image_encoder.parameters(), "lr": self.lr}],
            weight_decay=self.weight_decay,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)

    def training_step(self, batch, batch_idx):
        (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = self.get_loss(batch)
        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_indx):
        (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = self.get_loss(batch)

        output = {
            "loss": loss,
            "log": {
                "loss": loss,
                "counter_loss": counter_loss.sum(),
                "locs_loss": locs_loss.sum(),
                "galaxy_params_loss": galaxy_params_loss.sum(),
                "star_params_loss": star_params_loss.sum(),
                "galaxy_bool_loss": galaxy_bool_loss.sum(),
                "images": batch["images"],
                "locs": batch["locs"],
                "n_sources": batch["n_sources"],
            },
        }

        return output

    def make_validation_plots(self, outputs):
        # add some images to tensorboard for validating location/counts.
        # Only use 5 images in the last batch
        n_samples = min(5, len(outputs[-1]["log"]["n_sources"]))
        assert n_samples > 1
        true_n_sources = outputs[-1]["log"]["n_sources"][:n_samples]
        true_locs = outputs[-1]["log"]["locs"][:n_samples]
        images = outputs[-1]["log"]["images"][:n_samples]
        fig, axes = plt.subplots(
            nrows=n_samples,
            ncols=3,
            figsize=(
                20,
                20,
            ),
        )

        for i in range(n_samples):
            true_ax = axes[i, 0]
            recon_ax = axes[i, 1]
            res_ax = axes[i, 2]

            image = images[i]
            true_loc = true_locs[i]
            true_n_source = true_n_sources[i]

            with torch.no_grad():
                # get the estimated params
                self.image_encoder.eval()
                (
                    n_sources,
                    locs,
                    galaxy_params,
                    log_fluxes,
                    galaxy_bool,
                ) = self.image_encoder.map_estimate(image.unsqueeze(0))

            # draw true image
            assert len(image.shape) == 3
            image = image[0].cpu().numpy()  # first band.
            loc = locs[0].cpu().numpy()
            true_loc = true_loc.cpu().numpy()
            plotting.plot_image(fig, true_ax, image, true_loc, loc)
            true_ax.set_xlabel(
                f"True num: {true_n_source.item()}; Est num: {n_sources.item()}"
            )

            # only prediction/residual if at least 1 source.
            max_sources = n_sources.max().int().item()
            if max_sources > 0:
                assert max_sources == locs.shape[1]
                is_on_array = encoder.get_is_on_from_n_sources(n_sources, max_sources)
                star_bool = (1 - galaxy_bool) * is_on_array
                fluxes = log_fluxes.exp() * star_bool.unsqueeze(2)

                # draw reconstruction image.
                recon_image = self.dataset.image_decoder.render_images(
                    max_sources, n_sources, locs, galaxy_bool, galaxy_params, fluxes
                )

                assert len(locs.shape) == 3 and locs.size(0) == 1
                recon_image = recon_image[0, 0].cpu().numpy()
                res_image = (image - recon_image) / np.sqrt(image)

                # plot
                plotting.plot_image(fig, recon_ax, recon_image, loc)
                plotting.plot_image(fig, res_ax, res_image)

            else:
                slen = image.shape[0]
                plotting.plot_image(fig, recon_ax, np.zeros((slen, slen)))
                plotting.plot_image(fig, res_ax, np.zeros((slen, slen)))

        plt.subplots_adjust(hspace=0.25, wspace=-0.2)
        if self.logger:
            self.logger.experiment.add_figure(f"Val Images {self.current_epoch}", fig)
        plt.close(fig)

    def validation_epoch_end(self, outputs):

        # images for validation
        if self.current_epoch >= self.validation_plot_start:
            self.make_validation_plots(outputs)

        # log other losses
        # first we log some of the important losses and average over all batches.
        avg_loss = 0
        avg_counter_loss = 0
        avg_locs_loss = 0
        avg_galaxy_params_loss = 0
        avg_star_params_loss = 0
        avg_galaxy_bool_loss = 0

        tiles_per_batch = self.dataset.batch_size * self.image_encoder.n_tiles
        tiles_per_epoch = tiles_per_batch * self.dataset.n_batches

        # len(output) == n_batches
        # the sum below is over tiles_per_batch
        for output in outputs:
            avg_loss += output["loss"]
            avg_counter_loss += torch.sum(output["log"]["counter_loss"])
            avg_locs_loss += torch.sum(output["log"]["locs_loss"])
            avg_galaxy_params_loss += torch.sum(output["log"]["galaxy_params_loss"])
            avg_star_params_loss += torch.sum(output["log"]["star_params_loss"])
            avg_galaxy_bool_loss += torch.sum(output["log"]["galaxy_bool_loss"])

        avg_loss /= self.dataset.n_batches
        avg_counter_loss /= tiles_per_epoch
        avg_locs_loss /= tiles_per_epoch
        avg_galaxy_params_loss /= tiles_per_epoch
        avg_star_params_loss /= tiles_per_epoch
        avg_galaxy_bool_loss /= tiles_per_epoch

        logs = {
            "counter_loss": avg_counter_loss,
            "locs_loss": avg_locs_loss,
            "galaxy_params_loss": avg_galaxy_params_loss,
            "star_params_loss": avg_star_params_loss,
            "galaxy_bool_loss": avg_galaxy_bool_loss,
        }
        results = {"val_loss": avg_loss, "log": logs}
        return results

    @staticmethod
    def add_args(parser):

        # encoder parameters.
        parser.add_argument(
            "--ptile-slen",
            type=int,
            default=8,
            help="Side length of the padded tile in pixels.",
        )
        parser.add_argument(
            "--tile-slen",
            type=int,
            default=2,
            help="Distance between tile centers in pixels.",
        )

        parser.add_argument(
            "--max-detections",
            type=int,
            default=2,
            help="Number of max detections in each tile. ",
        )
        parser.add_argument(
            "--n-galaxy-params",
            type=int,
            default=8,
            help="Same as latent dim for galaxies.",
        )

        # network parameters.
        parser.add_argument("--enc-conv-c", type=int, default=20)
        parser.add_argument("--enc-kern", type=int, default=3)
        parser.add_argument("--enc-hidden", type=int, default=256)
        parser.add_argument("--momentum", type=float, default=0.5)

        # validation
        parser.add_argument("--validation-plot-start", type=int, default=5)

    @classmethod
    def from_args(cls, args, dataset):
        args_dict = vars(args)
        assert args_dict["latent_dim"] == args_dict["n_galaxy_params"]

        encoder_params = inspect.signature(encoder.ImageEncoder).parameters
        encoder_kwargs = {
            param: value
            for param, value in args_dict.items()
            if param in encoder_params
        }

        sleep_params = list(inspect.signature(cls).parameters)
        sleep_params.remove("dataset")
        sleep_params.remove("encoder_kwargs")
        sleep_kwargs = {
            param: value for param, value in args_dict.items() if param in sleep_params
        }

        return cls(dataset, encoder_kwargs, **sleep_kwargs)


class Objective(object):
    def __init__(self, dataset, encoder_kwargs):
        self.dataset = dataset
        self.encoder_kwargs = encoder_kwargs

    def __call__(self, trial):
        # suggested hyperparameters
        learning_rate = trial.loguniform("learning_rate", 1e-4, 1e-1)
        weight_decay = trial.loguniform("learning_rate", 1e-4, 1e-1)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"),
            monitor="val_loss",
        )

        metrics_callback = MetricsCallback()
        model = SleepPhase(
            self.dataset, self.encoder_kwargs, learning_rate, weight_decay
        )

        trainer = pl.trainer(
            checkpoint_callback=checkpoint_callback,
            callbacks=[metrics_callback],
            early_stop_callback=PyTorchLightningPruningCallback(
                trial, monitor="val_loss"
            ),
        )

        trainer.fit(model)
