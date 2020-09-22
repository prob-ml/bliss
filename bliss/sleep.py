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

from . import device, plotting
from .models import encoder

from optuna.integration import PyTorchLightningPruningCallback


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
        self.image_decoder = self.dataset.image_decoder
        self.image_encoder = encoder.ImageEncoder(**encoder_kwargs)

        # avoid calculating gradients of decoder.
        self.image_decoder.requires_grad_(False)

        self.validation_plot_start = validation_plot_start
        assert self.dataset.latent_dim == self.image_encoder.n_galaxy_params

        self.hparams = {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": self.dataset.batch_size,
            "n_batches": self.dataset.n_batches,
            "n_bands": self.dataset.n_bands,
            "max_sources_per_tile": self.image_decoder.max_sources_per_tile,
            "mean_sources_per_tile": self.image_decoder.mean_sources_per_tile,
            "min_sources_per_tile": self.image_decoder.min_sources_per_tile,
            "prob_galaxy": self.image_decoder.prob_galaxy,
        }

    def forward(self, image_ptiles, n_sources):
        return self.image_encoder(image_ptiles, n_sources)

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

        # flatten so first dimension is ptile
        batch_size = images.shape[0]
        n_tiles_per_image = self.image_decoder.n_tiles_per_image
        n_tiles = batch_size * n_tiles_per_image
        max_sources_per_tile = self.image_decoder.max_sources_per_tile
        n_bands = self.image_decoder.n_bands
        latent_dim = self.image_decoder.latent_dim

        true_tile_locs = true_tile_locs.view(n_tiles, max_sources_per_tile, 2)
        true_tile_galaxy_params = true_tile_galaxy_params.view(
            n_tiles, max_sources_per_tile, latent_dim
        )
        true_tile_log_fluxes = true_tile_log_fluxes.view(
            n_tiles, max_sources_per_tile, n_bands
        )
        true_tile_galaxy_bool = true_tile_galaxy_bool.view(
            n_tiles, max_sources_per_tile
        )
        true_tile_n_sources = true_tile_n_sources.flatten()
        true_tile_is_on_array = encoder.get_is_on_from_n_sources(
            true_tile_n_sources, max_sources_per_tile
        )

        # extract image tiles
        # true_tile_locs has shape = (n_ptiles x max_detections x 2)
        # true_tile_n_sources has shape = (n_ptiles)
        image_ptiles = self.image_encoder.get_images_in_tiles(images)
        n_ptiles = true_tile_is_on_array.size(0)
        max_detections = true_tile_is_on_array.size(1)

        pred = self(image_ptiles, true_tile_n_sources)

        # TODO: make .forward() and .get_params_in_tiles() just return correct dimensions ?
        prob_galaxy = pred["prob_galaxy"].view(n_ptiles, max_detections)
        true_tile_galaxy_bool = true_tile_galaxy_bool.view(n_ptiles, max_detections)

        # the loss for estimating the true number of sources
        n_source_log_probs = pred["n_source_log_probs"]
        true_tile_n_sources = true_tile_is_on_array.sum(1).long()  # per tile.
        one_hot_encoding = functional.one_hot(true_tile_n_sources, max_detections + 1)
        counter_loss = self._get_categorical_loss(n_source_log_probs, one_hot_encoding)

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
        galaxy_params_log_probs_all = self._get_params_logprob_all_combs(
            true_tile_galaxy_params,
            pred["galaxy_param_mean"],
            pred["galaxy_param_logvar"],
        )
        star_params_log_probs_all = self._get_params_logprob_all_combs(
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
            [{"params": self.image_encoder.parameters(), "lr": self.hparams.lr}],
            weight_decay=self.hparams.weight_decay,
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
                "galaxy_bool": batch["galaxy_bool"],
            },
        }

        return output

    def make_validation_plots(self, outputs):
        # add some images to tensorboard for validating location/counts.
        n_samples = min(10, len(outputs[-1]["log"]["n_sources"]))
        assert n_samples > 1

        # these are per tile
        true_n_sources_on_tiles = outputs[-1]["log"]["n_sources"][:n_samples]
        true_locs_on_tiles = outputs[-1]["log"]["locs"][:n_samples]
        true_galaxy_bools_on_tiles = outputs[-1]["log"]["galaxy_bool"][:n_samples]
        images = outputs[-1]["log"]["images"][:n_samples]

        # convert to full image parameters for plotting purposes.
        (
            true_n_sources,
            true_locs,
            true_galaxy_bools,
        ) = self.image_encoder.get_full_params_from_sampled_params(
            true_n_sources_on_tiles,
            true_locs_on_tiles,
            true_galaxy_bools_on_tiles.unsqueeze(-1),
        )

        figsize = (12, 4 * n_samples)
        fig, axes = plt.subplots(nrows=n_samples, ncols=3, figsize=figsize)

        for i in range(n_samples):
            true_ax = axes[i, 0]
            recon_ax = axes[i, 1]
            res_ax = axes[i, 2]

            image = images[None, i]
            slen = image.shape[-1]

            # true parameters on full image.
            true_loc = true_locs[None, i]
            true_n_source = true_n_sources[None, i]
            true_galaxy_bool = true_galaxy_bools[None, i].squeeze(-1)

            assert len(image.shape) == 4
            with torch.no_grad():
                # get the estimated params, these are *per tile*.
                self.image_encoder.eval()
                (
                    tile_n_sources,
                    tile_locs,
                    tile_galaxy_params,
                    tile_log_fluxes,
                    tile_galaxy_bool,
                ) = self.image_encoder.tiled_map_estimate(image)

            # convert tile estimates to full parameterization for plotting
            (
                n_sources,
                locs,
                galaxy_params,
                log_fluxes,
                galaxy_bool,
            ) = self.image_encoder.get_full_params_from_sampled_params(
                tile_n_sources,
                tile_locs,
                tile_galaxy_params,
                tile_log_fluxes,
                tile_galaxy_bool.unsqueeze(-1),
            )
            galaxy_bool = galaxy_bool.squeeze(-1)

            assert len(locs.shape) == 3 and locs.size(0) == 1
            assert locs.shape[1] == n_sources.max().int().item()

            # plot true image + number of sources first.
            image = image[0, 0].cpu().numpy()  # only first band.
            plotting.plot_image(fig, true_ax, image)
            true_ax.set_xlabel(
                f"True num: {true_n_source.item()}; Est num: {n_sources.item()}"
            )

            # continue only if at least one true source and predicted source.
            max_sources = true_loc.shape[1]
            if max_sources > 0 and n_sources.item() > 0:

                # draw reconstruction image.
                recon_image = self.image_decoder.render_images(
                    tile_n_sources,
                    tile_locs,
                    tile_galaxy_bool,
                    tile_galaxy_params,
                    tile_log_fluxes,
                )

                # round up true parameters.
                true_star_bool = self.image_decoder.get_star_bool(
                    true_n_source, true_galaxy_bool
                )
                true_galaxy_loc = self.image_decoder.get_galaxy_locs(
                    true_loc, true_galaxy_bool
                )
                true_star_loc = self.image_decoder.get_star_locs(
                    true_loc, true_star_bool
                )

                # round up estimated parameters.
                star_bool = self.image_decoder.get_star_bool(n_sources, galaxy_bool)
                galaxy_loc = self.image_decoder.get_galaxy_locs(locs, galaxy_bool)
                star_loc = self.image_decoder.get_star_locs(locs, star_bool)

                # convert everything to numpy + cpu so matplotlib can use it.
                true_galaxy_loc = true_galaxy_loc.cpu().numpy()[0]
                true_star_loc = true_star_loc.cpu().numpy()[0]
                galaxy_loc = galaxy_loc.cpu().numpy()[0]
                star_loc = star_loc.cpu().numpy()[0]

                recon_image = recon_image[0, 0].cpu().numpy()
                res_image = (image - recon_image) / np.sqrt(image)

                # plot and add locations.
                plotting.plot_image_locs(
                    true_ax, slen, true_galaxy_loc, galaxy_loc, colors=("r", "b")
                )
                plotting.plot_image_locs(
                    true_ax, slen, true_star_loc, star_loc, colors=("g", "m")
                )

                plotting.plot_image(fig, recon_ax, recon_image)
                plotting.plot_image_locs(
                    recon_ax, slen, galaxy_loc, star_loc, colors=("r", "b")
                )
                plotting.plot_image(fig, res_ax, res_image)

            else:
                slen = image.shape[0]
                plotting.plot_image(fig, recon_ax, np.zeros((slen, slen)))
                plotting.plot_image(fig, res_ax, np.zeros((slen, slen)))

        plt.subplots_adjust(hspace=0.2, wspace=0.4)
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

        tiles_per_batch = self.dataset.batch_size * self.image_encoder.n_tiles_per_image
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
            "val_loss": avg_loss,
            "counter_loss": avg_counter_loss,
            "locs_loss": avg_locs_loss,
            "galaxy_params_loss": avg_galaxy_params_loss,
            "star_params_loss": avg_star_params_loss,
            "galaxy_bool_loss": avg_galaxy_bool_loss,
        }
        results = {"val_loss": avg_loss, "log": logs}
        return results

    @staticmethod
    def _get_categorical_loss(n_source_log_probs, one_hot_encoding):
        assert torch.all(n_source_log_probs <= 0)
        assert n_source_log_probs.shape == one_hot_encoding.shape

        return -torch.sum(n_source_log_probs * one_hot_encoding, dim=1)

    @staticmethod
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

    @staticmethod
    def _get_kl_qp_galaxy_params(image_ptiles, galaxy_param_mean, galaxy_param_logvar):
        pass

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


class SleepObjective(object):
    def __init__(
        self,
        dataset,
        encoder_kwargs: dict,
        max_epochs: int,
        lr: tuple,
        weight_decay: tuple,
        model_dir,
        metrics_callback,
        monitor,
        gpus=0,
    ):
        self.dataset = dataset

        assert type(encoder_kwargs["enc_conv_c"]) is tuple
        assert type(encoder_kwargs["enc_hidden"]) is tuple
        assert (
            len(encoder_kwargs["enc_conv_c"]) == 3
            and len(encoder_kwargs["enc_hidden"]) == 3
        )
        self.encoder_kwargs = encoder_kwargs
        self.enc_conv_c_min = self.encoder_kwargs["enc_conv_c"][0]
        self.enc_conv_c_max = self.encoder_kwargs["enc_conv_c"][1]
        self.enc_conv_c_int = self.encoder_kwargs["enc_conv_c"][2]

        self.enc_hidden_min = self.encoder_kwargs["enc_hidden"][0]
        self.enc_hidden_max = self.encoder_kwargs["enc_hidden"][1]
        self.enc_hidden_int = self.encoder_kwargs["enc_hidden"][2]

        assert type(lr) is tuple
        assert type(weight_decay) is tuple
        assert len(lr) == 2 and len(weight_decay) == 2
        self.lr = lr
        self.weight_decay = weight_decay

        self.max_epochs = max_epochs
        self.model_dir = model_dir
        self.metrics_callback = metrics_callback
        self.monitor = monitor
        self.gpus = gpus

    def __call__(self, trial):
        self.encoder_kwargs["enc_conv_c"] = trial.suggest_int(
            "enc_conv_c",
            self.enc_conv_c_min,
            self.enc_conv_c_max,
            self.enc_conv_c_int,
        )

        self.encoder_kwargs["enc_hidden"] = trial.suggest_int(
            "enc_hidden",
            self.enc_hidden_min,
            self.enc_hidden_max,
            self.enc_hidden_int,
        )

        lr = trial.suggest_loguniform("learning rate", self.lr[0], self.lr[1])
        weight_decay = trial.suggest_loguniform(
            "weight_decay", self.weight_decay[0], self.weight_decay[1]
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            self.model_dir.joinpath("trial_{}".format(trial.number), "{epoch}"),
            monitor="val_loss",
        )

        model = SleepPhase(self.dataset, self.encoder_kwargs, lr, weight_decay).to(
            device
        )

        trainer = pl.Trainer(
            logger=False,
            gpus=self.gpus,
            checkpoint_callback=checkpoint_callback,
            max_epochs=self.max_epochs,
            callbacks=[self.metrics_callback],
            early_stop_callback=PyTorchLightningPruningCallback(
                trial, monitor=self.monitor
            ),
        )

        trainer.fit(model)

        return self.metrics_callback.metrics[-1][self.monitor].item()
