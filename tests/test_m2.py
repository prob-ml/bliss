import torch
import pytest
import os
import numpy as np

torch.manual_seed(84)
np.random.seed(43)

from bliss.datasets import sdss

@pytest.fixture(scope="module")
def trained_star_encoder_m2(sleep_setup, devices):
    overrides = dict(
        model="m2",
        dataset="m2" if devices.use_cuda else "cpu",
        training="m2" if devices.use_cuda else "cpu",
        optimizer="m2",
    )

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
        slen = test_image.shape[-1]

        # the true parameters
        true_locs = torch.from_numpy(hubble_data["true_locs"]).to(device)
        true_fluxes = torch.from_numpy(hubble_data["true_fluxes"]).to(device)
        nelec_per_nmgy = torch.from_numpy(hubble_data["nelec_per_nmgy"]).to(device)

        # get estimated parameters
        estimate = trained_star_encoder_m2.map_estimate(test_image.to(device))

        # check metrics if cuda is true
        if not devices.use_cuda:
            return

        # summary statistics
        sleep_tpr, sleep_ppv = sdss.get_summary_stats(
            estimate["locs"][0],
            true_locs,
            slen,
            estimate["fluxes"][0, :, 0],
            true_fluxes[:, 0],
            nelec_per_nmgy,
        )[0:2]

        print("Sleep phase TPR: ", sleep_tpr)
        print("Sleep phase PPV: ", sleep_ppv)

        assert sleep_tpr > 0.4
        assert sleep_ppv > 0.4
