import pytest
import json
import torch
import numpy as np
from torch import device

from celeste.utils import const
from celeste import sleep
from celeste import psf_transform
from celeste.datasets import simulated_datasets
from celeste.models import sourcenet_lib


# TODO: Set up pytest.mark.slow to enble skip on slow test


def pytest_addoption(parser):
    """
    Add --runslow option on command line for this project
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """
    Add slow markers to pytest
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# set up global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestStarSleepEncoder:
    # TODO: train the star encoder in sleep-face

    @pytest.mark.slow
    def test_galaxy_sleep(self):
        # setup train dataset
        # load star parameters
        param_file = const.data_path.joinpath("/data/default_star_parameters.json")
        with open(param_file, "r") as fp:
            data_params = json.load(fp)

        # make a smaller image
        data_params["max_stars"] = 40
        data_params["mean_stars"] = 20
        data_params["slen"] = 50

        # load psf
        psf_file = const.data_path.joinpath("/data/fitted_powerlaw_psf_params.npy")
        psf_params = torch.tensor(np.load(psf_file), device=device)
        power_law_psf = psf_transform.PowerLawPSF(psf_params)
        psf = power_law_psf.forward().detach()

        # set background
        background = torch.zeros(
            data_params["n_bands"],
            data_params["slen"],
            data_params["slen"],
            device=device,
        )
        background[0] = 686.0
        background[1] = 1123.0

        # simulate dataset
        n_images = 100
        star_dataset = simulated_datasets.StarsDataset.load_dataset_from_params(
            n_images,
            data_params,
            psf,
            background,
            transpose_psf=False,
            add_noise=True,
            draw_poisson=True,
        )

        # save images and star parameters to data folder
        batches = star_dataset.get_batch()

        with open(const.data_path.joinpath("star_sleep_test_data.json"), "w") as sf:
            sf.write(json.dumps(batches))

        # setup encoder
        # train encoder on 100*100 images
        # How to initialize the log_fluxes?
        log_fluxes = None
        star_encoder = sourcenet_lib.SourceEncoder(
            slen=100,
            n_bands=1,
            ptile_slen=20,
            step=5,
            edge_padding=5,
            max_detections=2,
            n_source_params=log_fluxes,
        ).to(const.device)

        # TODO: set up optimizer (torch.optim), "learning rate" and "weight_decay"
        #       and set up state_dict file path
        # Can we set default of optimizer to be Adam and let user to choose other types?
        # I'm also not sure if we should let user to choose weight_decay and lr

        # TODO: Train through StarSleep.run_sleep (loses), and get estimated parameters
        #       log_fluxes, locs and images (_get_params_from_data)

        # TODO: compare true locations with estimated locations (substract and check
        #       zeros)
        #       compare true images and estimated locations

        # The above part will be completed after the encoder tutorial
