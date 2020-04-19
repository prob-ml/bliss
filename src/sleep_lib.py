import math
import time
from itertools import permutations
import numpy as np
import torch

from .utils import const


def isnan(x):
    return x != x


#############################
# functions to get loss for training the counter
############################

def _get_categorical_loss(log_probs, one_hot_encoding):
    assert torch.all(log_probs <= 0)
    assert log_probs.shape[0] == one_hot_encoding.shape[0]
    assert log_probs.shape[1] == one_hot_encoding.shape[1]

    return torch.sum(
        -log_probs * one_hot_encoding, dim=1)


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
    locs_log_probs_all = const.eval_normal_logprob(_true_locs,
                                                   _loc_mean, _loc_log_var).sum(dim=3)

    return locs_log_probs_all


def _get_log_probs_all_perms(locs_log_probs_all, source_param_log_probs_all, is_on_array):
    max_detections = source_param_log_probs_all.shape[-1]
    batchsize = source_param_log_probs_all.shape[0]

    locs_loss_all_perm = torch.zeros(batchsize,
                                     math.factorial(max_detections)).cuda()
    source_param_loss_all_perm = torch.zeros(batchsize,
                                             math.factorial(max_detections)).cuda()
    i = 0
    for perm in permutations(range(max_detections)):
        locs_loss_all_perm[:, i] = \
            (locs_log_probs_all[:, perm, :].diagonal(dim1=1, dim2=2) *
             is_on_array).sum(1)

        source_param_loss_all_perm[:, i] = \
            (source_param_log_probs_all[:, perm].diagonal(dim1=1, dim2=2) *
             is_on_array).sum(1)
        i += 1

    return locs_loss_all_perm, source_param_loss_all_perm


def get_min_perm_loss(locs_log_probs_all, source_params_log_probs_all, is_on_array):
    locs_log_probs_all_perm, source_params_log_probs_all_perm = \
        _get_log_probs_all_perms(locs_log_probs_all, source_params_log_probs_all, is_on_array)

    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim=1)
    source_params_loss = -torch.gather(source_params_log_probs_all_perm, 1, indx.unsqueeze(1)).squeeze()

    return locs_loss, source_params_loss, indx


class SourceSleep(object):
    def __init__(self, encoder, dataset, n_epochs, n_source_params, out_filename,
                 optimizer=None, batchsize=32, print_every=10):
        """
        In this case, source_params means either flux or galaxy latent params.
        Args:
            encoder:
            dataset:
            optimizer:
            n_epochs:
            out_filename:
            batchsize:
            print_every:
        """

        self.optimizer = optimizer

        self.n_epochs = n_epochs
        self.out_filename = out_filename
        self.print_every = print_every

        self.encoder = encoder
        self.dataset = dataset
        self.batchsize = batchsize
        self.n_source_params = n_source_params

    def run_sleep(self):
        train_losses = np.zeros((4, self.n_epochs))

        for epoch in range(self.n_epochs):
            t0 = time.time()

            avg_loss, counter_loss, locs_loss, fluxes_loss = self.eval_sleep(train=True)

            elapsed = time.time() - t0
            print(
                f'{epoch} loss: {avg_loss:.4f}; counter loss: {counter_loss:.4f}; locs loss: {locs_loss:.4f};'
                f' fluxes loss: {fluxes_loss:.4f} \t [{elapsed:.1f} seconds]'
            )

            train_losses[:, epoch] = np.array([avg_loss, counter_loss, locs_loss, fluxes_loss])
            np.savetxt(f"{self.out_filename}-train_losses", train_losses)

            if (epoch % self.print_every) == 0:
                test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss = self.eval_sleep(train=False)

                print(
                    f'**** test loss: {test_loss:.3f}; counter loss: {test_counter_loss:.3f}; '
                    f'locs loss: {test_locs_loss:.3f}; fluxes loss: {test_fluxes_loss:.3f} ****'
                )

                print("writing the encoder parameters to " + self.out_filename.as_posix())
                torch.save(self.encoder.state_dict(), self.out_filename)

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
            true_source_params, true_locs, images = self._get_params_from_data(data)

            if train:
                self.encoder.train()
                self.optimizer.zero_grad()
            else:
                self.encoder.eval()

            # evaluate log q
            loss, counter_loss, locs_loss, source_params_loss = \
                self._get_inv_kl_loss(images, true_locs, true_source_params)[0:4]

            if train:
                loss.backward()
                self.optimizer.step()

            avg_loss += loss.item() * images.shape[0] / len(self.dataset)
            avg_counter_loss += counter_loss.sum().item() / (len(self.dataset) * self.encoder.n_patches)
            avg_source_params_loss += source_params_loss.sum().item() / (len(self.dataset) * self.encoder.n_patches)
            avg_locs_loss += locs_loss.sum().item() / (len(self.dataset) * self.encoder.n_patches)

        return avg_loss, avg_counter_loss, avg_locs_loss, avg_source_params_loss

    # TODO: Fix misnomer variables without redundant code (see docstring)?
    def _get_inv_kl_loss(self, images, true_locs, true_source_params):
        """
        In the case of stars, this function has some misnomer variables:
        * true_source_params = true_fluxes
        * source_param_mean, source_param_logvar = log_flux_mean, log_flux_logvar.

        Be careful!

        Args:
            images:
            true_locs:
            true_source_params:

        Returns:

        """
        # true_source_params are either fluxes or galaxy_params.
        # extract image patches
        image_patches, true_patch_locs, true_patch_source_params, true_patch_n_sources, true_patch_is_on_array = \
            self.encoder.get_image_patches(images, true_locs, true_source_params,
                                           clip_max_sources=True)

        loc_mean, loc_logvar, source_param_mean, source_param_logvar, log_probs_n_sources_patch = \
            self.encoder.forward(image_patches, n_sources=true_patch_n_sources)

        loss, counter_loss, locs_loss, fluxes_loss, perm_indx = \
            self._get_params_loss(loc_mean, loc_logvar,
                                  source_param_mean, source_param_logvar, log_probs_n_sources_patch,
                                  true_patch_locs, true_patch_source_params,
                                  true_patch_is_on_array.float())

        return loss, counter_loss, locs_loss, fluxes_loss, perm_indx, log_probs_n_sources_patch

    def _get_params_loss(self, loc_mean, loc_logvar,
                         source_param_mean, source_param_logvar, log_probs,
                         true_locs, true_source_params, true_is_on_array):
        """
        In the case of stars, this function has some misnomer variables:
        * true_source_params = true_fluxes
        * source_param_mean, source_param_logvar = log_flux_mean, log_flux_logvar.

        Args:
            loc_mean:
            loc_logvar:
            source_param_mean:
            source_param_logvar:
            log_probs:
            true_locs:
            true_source_params:
            true_is_on_array:

        Returns:

        """
        # this is batchsize x (max_stars x max_detections)
        # max_detections = log_flux_mean.shape[1]
        # the log prob for each observed location x mean
        locs_log_probs_all = _get_locs_logprob_all_combs(true_locs, loc_mean, loc_logvar)

        source_param_log_probs_all = self._get_source_params_logprob_all_combs(true_source_params,
                                                                               source_param_mean,
                                                                               source_param_logvar)

        locs_loss, source_param_loss, perm_indx = get_min_perm_loss(locs_log_probs_all, source_param_log_probs_all,
                                                                    true_is_on_array)

        true_n_stars = true_is_on_array.sum(1)
        one_hot_encoding = const.get_one_hot_encoding_from_int(true_n_stars, log_probs.shape[1])
        counter_loss = _get_categorical_loss(log_probs, one_hot_encoding)

        loss_vec = (locs_loss * (locs_loss.detach() < 1e6).float() + source_param_loss + counter_loss)

        loss = loss_vec.mean()

        return loss, counter_loss, locs_loss, source_param_loss, perm_indx

    def _get_transformed_source_params(self, true_source_params, source_param_mean, source_param_logvar):
        _true_source_params = true_source_params.view(self.batchsize, true_source_params.shape[1], 1,
                                                      self.n_source_params)
        _log_source_param_mean = source_param_mean.view(self.batchsize, 1, source_param_mean.shape[1],
                                                        self.n_source_params)
        _log_source_param_log_var = source_param_logvar.view(self.batchsize, 1, source_param_mean.shape[1],
                                                             self.n_source_params)

        return _true_source_params, _log_source_param_mean, _log_source_param_log_var

    def _get_source_params_logprob_all_combs(self, true_gal_params, gal_param_mean, gal_param_logvar):
        return torch.zeros(0)

    @staticmethod
    def _get_params_from_data(data):
        return torch.zeros(0)


class StarSleep(SourceSleep):

    @staticmethod
    def _get_params_from_data(data):
        return data['fluxes'], data['locs'], data['images']

    def _get_source_params_logprob_all_combs(self, true_fluxes, log_flux_mean, log_flux_logvar):
        _true_fluxes, _log_flux_mean, _log_flux_logvar = self._get_transformed_source_params(true_fluxes, log_flux_mean,
                                                                                             log_flux_logvar)

        # this is batchsize x (max_stars x max_detections)
        # the log prob for each observed location x mean
        flux_log_probs_all = const.eval_lognormal_logprob(_true_fluxes,
                                                          _log_flux_mean, _log_flux_logvar).sum(dim=3)
        return flux_log_probs_all


class GalaxySleep(SourceSleep):

    @staticmethod
    def _get_params_from_data(data):
        return data['gal_params'], data['locs'], data['images']

    def _get_source_params_logprob_all_combs(self, true_gal_params, gal_param_mean, gal_param_logvar):
        _true_gal_params, _gal_param_mean, _gal_param_logvar = self._get_transformed_source_params(true_gal_params,
                                                                                                   gal_param_mean,
                                                                                                   gal_param_logvar)
        gal_param_log_probs_all = const.eval_normal_logprob(_true_gal_params,
                                                            _gal_param_mean,
                                                            _gal_param_logvar).sum(dim=3)
        return gal_param_log_probs_all
