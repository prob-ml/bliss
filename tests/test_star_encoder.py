import pytest
import json
import torch
import numpy as np

from celeste import utils
from celeste import train
from celeste import psf_transform
from celeste.datasets import simulated_datasets
from celeste.models import sourcenet_lib


class TestStarSleepEncoder:
    # TODO: train the star encoder in sleep-face

    def test_star_sleep(self):
        # setup Star Encoder
        star_encoder = sourcenet_lib.SourceEncoder(
            slen=50,
            ptile_slen=8,
            step=2,
            edge_padding=3,
            n_bands=2,
            max_detections=2,
            n_source_params=2,
        ).to(utils.device)

        # create training dataset
        param_file = utils.data_path.joinpath("default_star_parameters.json")
        with open(param_file, "r") as fp:
            data_params = json.load(fp)

        # make a smaller image
        data_params["max_stars"] = 15
        data_params["mean_stars"] = 8
        data_params["min_stars"] = 0
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
        n_images = 192
        star_dataset = simulated_datasets.StarDataset.load_dataset_from_params(
            n_images,
            data_params,
            psf,
            background,
            transpose_psf=False,
            add_noise=True,
            draw_poisson=True,
        )

        # train encoder
        # training wrapper
        StarSleepTrain = train.SleepTraining(
            model=star_encoder,
            dataset=star_dataset,
            slen=50,
            num_bands=2,
            n_source_params=2,
            out_name="star_encoder_sleepTrain",
            verbose=True,
        )

        StarSleepTrain.run(n_epochs=30)
