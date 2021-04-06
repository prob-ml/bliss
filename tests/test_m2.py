import os

import torch
import pytest
import numpy as np

from bliss import metrics as metrics_lib

torch.manual_seed(841)
np.random.seed(431)


@pytest.fixture(scope="module")
def trained_star_encoder_m2(sleep_setup, devices):
    overrides = {
        "model": "m2",
        "dataset": "m2" if devices.use_cuda else "cpu",
        "training": "m2" if devices.use_cuda else "cpu",
        "training.trainer.check_val_every_n_epoch": 9999,
        "training.n_epochs": 50 if devices.use_cuda else 1,
        "optimizer": "m2",
    }

    sleep_net = sleep_setup.get_trained_sleep(overrides)
    return sleep_net.image_encoder.to(devices.device)


class TestStarSleepEncoderM2:
    def test_star_sleep_m2(self, trained_star_encoder_m2, devices, paths):
        device = devices.device

        # the trained star encoder
        trained_star_encoder_m2.eval()

        # load hubble parameters and SDSS image
        hubble_data = np.load(os.path.join(paths["data"], "true_hubble_m2.npz"))

        # the SDSS image
        test_image = torch.from_numpy(hubble_data["sdss_image"]).unsqueeze(0).to(device)
        slen = 100

        # the true parameters
        true_locs = torch.from_numpy(hubble_data["true_locs"]).to(device)
        true_fluxes = torch.from_numpy(hubble_data["true_fluxes"]).to(device)

        # get estimated parameters
        estimate = trained_star_encoder_m2.map_estimate(test_image.to(device), slen)

        # check metrics if cuda is true
        if not devices.use_cuda:
            return

        # summary statistics
        sleep_tpr, sleep_ppv = metrics_lib.get_tpr_ppv(
            true_locs * slen,
            2.5 * torch.log10(true_fluxes[:, 0:1]),
            estimate["locs"][0] * slen,
            2.5 * torch.log10(estimate["fluxes"][0, :, 0:1]),
            slack=0.5,
        )

        print("Sleep phase TPR: ", sleep_tpr)
        print("Sleep phase PPV: ", sleep_ppv)

        assert sleep_tpr > 0.4
        assert sleep_ppv > 0.4
