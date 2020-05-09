import math
import time
from itertools import permutations
import numpy as np
import torch
from abc import ABC, abstractmethod

from .utils import const


#############################
# functions to get loss for training the counter
############################


def _get_categorical_loss(n_source_log_probs, one_hot_encoding):
    assert torch.all(n_source_log_probs <= 0)
    assert n_source_log_probs.shape[0] == one_hot_encoding.shape[0]
    assert n_source_log_probs.shape[1] == one_hot_encoding.shape[1]

    return torch.sum(-n_source_log_probs * one_hot_encoding, dim=1)


def _permute_losses_mat(losses_mat, perm):
    batchsize = losses_mat.shape[0]
    max_stars = losses_mat.shape[1]

    assert perm.shape[0] == batchsize
    assert perm.shape[1] == max_stars

    return torch.gather(losses_mat, 2, perm.unsqueeze(2)).squeeze()


def _get_locs_logprob_all_combs(true_locs, loc_mean, loc_log_var):
    batchsize = true_locs.shape[0]

    # get losses for locations
    _loc_mean = loc_mean.view(batchsize, 1, loc_mean.shape[1], 2)
    _loc_log_var = loc_log_var.view(batchsize, 1, loc_mean.shape[1], 2)
    _true_locs = true_locs.view(batchsize, true_locs.shape[1], 1, 2)

    # this will return a large error if star is off
    _true_locs = _true_locs + (_true_locs == 0).float() * 1e16

    # this is batchsize x (max_stars x max_detections)
    # the log prob for each observed location x mean
    locs_log_probs_all = const.eval_normal_logprob(
        _true_locs, _loc_mean, _loc_log_var
    ).sum(dim=3)

    return locs_log_probs_all


def _get_log_probs_all_perms(
    locs_log_probs_all, source_param_log_probs_all, is_on_array
):
    max_detections = source_param_log_probs_all.shape[-1]
    batchsize = source_param_log_probs_all.shape[0]

    locs_loss_all_perm = torch.zeros(batchsize, math.factorial(max_detections)).cuda()
    source_param_loss_all_perm = torch.zeros(
        batchsize, math.factorial(max_detections)
    ).cuda()
    i = 0
    for perm in permutations(range(max_detections)):
        locs_loss_all_perm[:, i] = (
            locs_log_probs_all[:, perm, :].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)

        source_param_loss_all_perm[:, i] = (
            source_param_log_probs_all[:, perm].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)
        i += 1

    return locs_loss_all_perm, source_param_loss_all_perm


def _get_min_perm_loss(locs_log_probs_all, source_params_log_probs_all, is_on_array):
    (
        locs_log_probs_all_perm,
        source_params_log_probs_all_perm,
    ) = _get_log_probs_all_perms(
        locs_log_probs_all, source_params_log_probs_all, is_on_array
    )

    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim=1)
    source_params_loss = -torch.gather(
        source_params_log_probs_all_perm, 1, indx.unsqueeze(1)
    ).squeeze()

    return locs_loss, source_params_loss, indx


class SourceSleep(ABC):
    def __init__(
        self,
        encoder,
        dataset,
        n_epochs,
        n_source_params,
        state_dict_file=None,
        output_file=None,
        optimizer=None,
        batchsize=32,
        print_every=20,
    ):
        """
        In this case, source_params means either flux or galaxy latent params.
        Args:
            encoder:
            dataset:
            optimizer:
            n_epochs:
            batchsize:
            print_every:
        """

        self.optimizer = optimizer
        self.n_epochs = n_epochs

        self.encoder = encoder
        self.dataset = dataset
        self.batchsize = batchsize
        self.n_source_params = n_source_params

        # logging.
        self.print_every = print_every
        self.state_dict_file = state_dict_file  # paths.
        self.output_file = output_file
        self.train_losses_file = output_file.parent.joinpath("train_losses.txt")

        assert (
            len(self.dataset) % self.batchsize == 0
        ), "For simplicity, make length of dataset divisible by batchsize."

    def _log_train(
        self,
        epoch,
        avg_loss,
        counter_loss,
        locs_loss,
        source_param_loss,
        train_losses,
        t0,
    ):

        # print and save test results.
        elapsed = time.time() - t0
        out_text = (
            f"{epoch} loss: {avg_loss:.4f}; counter loss: {counter_loss:.4f}; locs loss: {locs_loss:.4f}; "
            f"source_params loss: {source_param_loss:.4f} \t [{elapsed:.1f} seconds]"
        )
        print(out_text)
        if self.output_file:
            with open(self.output_file, "a") as out:
                print(out_text, file=out)

        np.savetxt(self.train_losses_file, train_losses)

    def _log_eval(
        self, test_loss, test_counter_loss, test_locs_loss, test_source_param_loss
    ):

        out_text = (
            f"**** test loss: {test_loss:.3f}; counter loss: {test_counter_loss:.3f}; "
            f"locs loss: {test_locs_loss:.3f}; source param loss: {test_source_param_loss:.3f} ****"
        )
        print(out_text)
        if self.output_file:
            with open(self.output_file, "a") as out:
                print(out_text, file=out)

        print("writing the encoder parameters to " + self.state_dict_file.as_posix())
        torch.save(self.encoder.state_dict(), self.state_dict_file)

    def run_sleep(self):
        assert self.optimizer is not None and self.state_dict_file is not None

        train_losses = np.zeros((4, self.n_epochs))

        for epoch in range(self.n_epochs):
            t0 = time.time()
            avg_loss, counter_loss, locs_loss, source_param_loss = self.eval_sleep(
                train=True
            )
            train_losses[:, epoch] = np.array(
                [avg_loss, counter_loss, locs_loss, source_param_loss]
            )
            self._log_train(
                epoch,
                avg_loss,
                counter_loss,
                locs_loss,
                source_param_loss,
                train_losses,
                t0,
            )

            if (epoch % self.print_every) == 0:
                (
                    test_loss,
                    test_counter_loss,
                    test_locs_loss,
                    test_source_param_loss,
                ) = self.eval_sleep(train=False)
                self._log_eval(
                    test_loss, test_counter_loss, test_locs_loss, test_source_param_loss
                )

    def eval_sleep(self, train=False):
        assert not train or self.optimizer, "For training you need an optimizer. "

        avg_loss = 0.0
        avg_counter_loss = 0.0
        avg_locs_loss = 0.0
        avg_source_params_loss = 0.0

        num_batches = int(len(self.dataset) / self.batchsize)

        for i in range(num_batches):
            data = self.dataset.get_batch(batchsize=self.batchsize)

            # fluxes or gal_params are returned as true_source_params.
            # already in cuda.
            true_source_params, true_locs, images = self._get_params_from_data(data)

            if train:
                self.encoder.train()
                self.optimizer.zero_grad()
            else:
                self.encoder.eval()

            # evaluate log q
            loss, counter_loss, locs_loss, source_params_loss = self._get_inv_kl_loss(
                images, true_locs, true_source_params
            )[0:4]

            if train:
                loss.backward()
                self.optimizer.step()

            avg_loss += loss.item() * images.shape[0] / len(self.dataset)
            avg_counter_loss += counter_loss.sum().item() / (
                len(self.dataset) * self.encoder.n_tiles
            )
            avg_source_params_loss += source_params_loss.sum().item() / (
                len(self.dataset) * self.encoder.n_tiles
            )
            avg_locs_loss += locs_loss.sum().item() / (
                len(self.dataset) * self.encoder.n_tiles
            )

        return avg_loss, avg_counter_loss, avg_locs_loss, avg_source_params_loss

    def _get_inv_kl_loss(self, images, true_locs, true_source_params):
        """
        NOTE: true_source_params are either log_fluxes or galaxy_params (both are normal unconstrained
        normal variables).

        Args:
            images:
            true_locs:
            true_source_params:

        Returns:

        """
        # extract image tiles
        (
            image_ptiles,
            true_tile_locs,
            true_tile_source_params,
            true_tile_n_sources,
            true_tile_is_on_array,
        ) = self.encoder.get_image_ptiles(
            images, true_locs, true_source_params, clip_max_sources=True
        )

        (
            loc_mean,
            loc_logvar,
            source_param_mean,
            source_param_logvar,
            log_probs_n_sources_per_tile,
        ) = self.encoder.forward(image_ptiles, n_sources=true_tile_n_sources)

        (
            loss,
            counter_loss,
            locs_loss,
            source_param_loss,
            perm_indx,
        ) = self._get_params_loss(
            loc_mean,
            loc_logvar,
            source_param_mean,
            source_param_logvar,
            log_probs_n_sources_per_tile,
            true_tile_locs,
            true_tile_source_params,
            true_tile_is_on_array.float(),
        )

        return (
            loss,
            counter_loss,
            locs_loss,
            source_param_loss,
            perm_indx,
            log_probs_n_sources_per_tile,
        )

    def _get_params_loss(
        self,
        loc_mean,
        loc_logvar,
        source_param_mean,
        source_param_logvar,
        n_source_log_probs,
        true_locs,
        true_source_params,
        true_is_on_array,
    ):
        """
        NOTE: All the quantities are per-tile quantities on first dimension, for simplicity not added
        to names.

        Args:
            loc_mean:
            loc_logvar:
            source_param_mean:
            source_param_logvar:
            n_source_log_probs:
            true_locs:
            true_source_params:
            true_is_on_array:

        Returns:

        """
        # this is batchsize x (max_stars x max_detections)
        # max_detections = log_flux_mean.shape[1]
        # the log prob for each observed location x mean
        locs_log_probs_all = _get_locs_logprob_all_combs(
            true_locs, loc_mean, loc_logvar
        )

        source_param_log_probs_all = self._get_source_params_logprob_all_combs(
            true_source_params, source_param_mean, source_param_logvar
        )

        locs_loss, source_param_loss, perm_indx = _get_min_perm_loss(
            locs_log_probs_all, source_param_log_probs_all, true_is_on_array
        )

        true_n_stars = true_is_on_array.sum(1)
        one_hot_encoding = const.get_one_hot_encoding_from_int(
            true_n_stars, n_source_log_probs.shape[1]
        )
        counter_loss = _get_categorical_loss(n_source_log_probs, one_hot_encoding)

        loss_vec = (
            locs_loss * (locs_loss.detach() < 1e6).float()
            + source_param_loss
            + counter_loss
        )

        loss = loss_vec.mean()

        return loss, counter_loss, locs_loss, source_param_loss, perm_indx

    def _get_transformed_source_params(
        self, true_source_params, source_param_mean, source_param_logvar
    ):
        n_tiles = true_source_params.shape[0]

        _true_source_params = true_source_params.view(
            n_tiles, true_source_params.shape[1], 1, self.n_source_params
        )
        _source_param_mean = source_param_mean.view(
            n_tiles, 1, source_param_mean.shape[1], self.n_source_params
        )
        _source_param_logvar = source_param_logvar.view(
            n_tiles, 1, source_param_mean.shape[1], self.n_source_params
        )

        return _true_source_params, _source_param_mean, _source_param_logvar

    def _get_source_params_logprob_all_combs(
        self, true_source_params, source_param_mean, source_param_logvar
    ):
        (
            _true_source_params,
            _source_param_mean,
            _source_param_logvar,
        ) = self._get_transformed_source_params(
            true_source_params, source_param_mean, source_param_logvar
        )
        source_param_log_probs_all = const.eval_normal_logprob(
            _true_source_params, _source_param_mean, _source_param_logvar
        ).sum(dim=3)
        return source_param_log_probs_all

    @staticmethod
    @abstractmethod
    def _get_params_from_data(data):
        pass


class StarSleep(SourceSleep):
    @staticmethod
    def _get_params_from_data(data):
        return data["log_fluxes"], data["locs"], data["images"]


class GalaxySleep(SourceSleep):
    @staticmethod
    def _get_params_from_data(data):
        return data["gal_params"], data["locs"], data["images"]
