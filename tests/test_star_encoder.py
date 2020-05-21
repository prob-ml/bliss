import pytest
import json
import torch
import numpy as np

from celeste import utils
from celeste import train
from celeste import psf_transform
from celeste.datasets import simulated_datasets
from celeste.models import sourcenet


class TestStarSleepEncoder:
    @pytest.mark.parametrize(
        "n_star", [1, 3],
    )
    def test_star_sleep(self, n_star):

        # create training dataset
        param_file = utils.config_path.joinpath(
            "dataset_params/default_star_parameters.json"
        )
        with open(param_file, "r") as fp:
            data_params = json.load(fp)

        data_params["max_stars"] = 50
        data_params["mean_stars"] = 40
        data_params["min_stars"] = 30
        data_params["f_min"] = 1e3
        data_params["slen"] = 50

        # load psf
        psf_file = utils.data_path.joinpath("fitted_powerlaw_psf_params.npy")
        psf_params = torch.tensor(np.load(psf_file), device=utils.device)
        power_law_psf = psf_transform.PowerLawPSF(psf_params)
        psf = power_law_psf.forward().detach()

        # set background
        background = torch.zeros(
            data_params["n_bands"],
            data_params["slen"],
            data_params["slen"],
            device=utils.device,
        )
        background[0] = 686.0
        background[1] = 1123.0

        # simulate dataset
        n_images = 128
        star_dataset = simulated_datasets.StarDataset.load_dataset_from_params(
            n_images,
            data_params,
            psf,
            background,
            transpose_psf=False,
            add_noise=True,
            draw_poisson=True,
        )

        # setup Star Encoder
        star_encoder = sourcenet.SourceEncoder(
            slen=data_params["slen"],
            ptile_slen=8,
            step=2,
            edge_padding=3,
            n_bands=2,
            max_detections=2,
            n_source_params=2,
        ).to(utils.device)

        # train encoder
        # training wrapper
        StarSleepTrain = train.SleepTraining(
            model=star_encoder,
            dataset=star_dataset,
            slen=data_params["slen"],
            num_bands=2,
            n_source_params=2,
            verbose=False,
            batchsize=32,
        )

        StarSleepTrain.run(n_epochs=50)

        # load test image
        test_star = torch.load(
            utils.data_path.joinpath(str(n_star) + "star_test_params")
        )
        test_image = test_star["images"]

        assert test_star["fluxes"].min() > 0

        # get the estimated params
        locs, source_params, n_sources = star_encoder.sample_encoder(
            test_image.to(utils.device),
            n_samples=1,
            return_map_n_sources=True,
            return_map_source_params=True,
        )

        # test that parameters match.
        assert n_sources == test_star["n_sources"].to(utils.device)
        assert (
            abs(
                (test_star["locs"].sort(1)[0].to(utils.device) - locs.sort(1)[0])
                * test_image.size(-1)
            ).max()
            <= 0.5
        )

        # fluxes
        diff = abs(
            test_star["log_fluxes"].sort(1)[0].to(utils.device)
            - source_params.sort(1)[0]
        )
        assert torch.all(diff <= source_params.min() * 0.10) and torch.all(
            diff <= test_star["log_fluxes"].min() * 0.10
        )
