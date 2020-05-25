import torch
import json
import pytest

from celeste.datasets import simulated_datasets
from celeste import device


# TODO: Test galaxy dataset and batchsize=1 separately.
class TestSDSSDataset:
    @pytest.fixture(scope="module")
    def data_params(self, config_path):
        params_file = config_path.joinpath(
            "dataset_params/default_star_parameters.json"
        )

        with open(params_file, "r") as fp:
            data_params = json.load(fp)

        data_params["slen"] = 50
        data_params["mean_stars"] = 40

        return data_params

    def test_fresh_data(self, data_params, fitted_powerlaw_psf):
        # this checks that we are actually drawing fresh data
        # at each epoch (or not)

        n_images = 120
        # set background
        background = torch.zeros(
            data_params["n_bands"],
            data_params["slen"],
            data_params["slen"],
            device=device,
        )
        background[0] = 686.0
        background[1] = 1123.0

        star_dataset = simulated_datasets.StarDataset.load_dataset_from_params(
            n_images=n_images,
            data_params=data_params,
            psf=fitted_powerlaw_psf,
            background=background,
            transpose_psf=False,
            add_noise=True,
        )

        #############################################
        # Check: by creating new batches fresh data
        # should be drawn
        #############################################
        num_epoch = 5
        images_vec = torch.zeros(num_epoch, device=device)
        locs_vec = torch.zeros(num_epoch, device=device)
        fluxes_vec = torch.zeros(num_epoch, device=device)
        n_sources_vec = torch.zeros(num_epoch, device=device)

        # get batch
        batchsize = 8
        num_batches = int(len(star_dataset) / batchsize)

        assert len(star_dataset) == n_images
        assert (
            len(star_dataset) >= num_epoch * batchsize * 2
        ), "Dataset repeats after n_images "

        for i in range(num_epoch):
            images_mean = 0
            true_locs_mean = 0
            true_fluxes_mean = 0
            true_n_sources_mean = 0

            for j in range(num_batches):
                batch = star_dataset.get_batch(batchsize)

                true_fluxes = batch["fluxes"]
                true_locs = batch["locs"]
                true_n_sources = batch["n_sources"]
                images = batch["images"]

                images_mean += images.mean()
                true_locs_mean += true_locs.mean()
                true_fluxes_mean += true_fluxes.mean()
                true_n_sources_mean += true_n_sources.float().mean()

            images_vec[i] = images_mean
            locs_vec[i] = true_locs_mean
            fluxes_vec[i] = true_fluxes_mean
            n_sources_vec[i] = true_n_sources_mean

        assert len(images_vec.unique()) == num_epoch
        assert len(locs_vec.unique()) == num_epoch
        assert len(fluxes_vec.unique()) == num_epoch
        assert len(n_sources_vec.unique()) == num_epoch
