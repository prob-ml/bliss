import torch
import pytest
import os
import numpy as np

torch.manual_seed(84)
np.random.seed(43)


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


def filter_params(locs, fluxes, slen, pad=5):
    assert len(locs.shape) == 2

    if fluxes is not None:
        assert len(fluxes.shape) == 1
        assert len(fluxes) == len(locs)

    _locs = locs * (slen - 1)
    which_params = (
        (_locs[:, 0] > pad)
        & (_locs[:, 0] < (slen - pad))
        & (_locs[:, 1] > pad)
        & (_locs[:, 1] < (slen - pad))
    )

    if fluxes is not None:
        return locs[which_params], fluxes[which_params]
    else:
        return locs[which_params], None


def get_locs_error(locs, true_locs):
    # get matrix of Linf error in locations
    # truth x estimated
    return torch.abs(locs.unsqueeze(0) - true_locs.unsqueeze(1)).max(2)[0]


def get_fluxes_error(fluxes, true_fluxes):
    # get matrix of l1 error in log flux
    # truth x estimated
    return torch.abs(
        torch.log10(fluxes).unsqueeze(0) - torch.log10(true_fluxes).unsqueeze(1)
    )


def get_mag_error(mags, true_mags):
    # get matrix of l1 error in magnitude
    # truth x estimated
    return torch.abs(mags.unsqueeze(0) - true_mags.unsqueeze(1))


def convert_nmgy_to_mag(nmgy):
    return 22.5 - 2.5 * torch.log10(nmgy)


def get_summary_stats(
    est_locs, true_locs, slen, est_fluxes, true_fluxes, nelec_per_nmgy, pad=5, slack=0.5
):
    est_locs, est_fluxes = filter_params(est_locs, est_fluxes, slen, pad)
    true_locs, true_fluxes = filter_params(true_locs, true_fluxes, slen, pad)

    est_mags = convert_nmgy_to_mag(est_fluxes / nelec_per_nmgy)
    true_mags = convert_nmgy_to_mag(true_fluxes / nelec_per_nmgy)
    mag_error = get_mag_error(est_mags, true_mags)

    locs_error = get_locs_error(est_locs * (slen - 1), true_locs * (slen - 1))
    tpr_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=1).float()
    ppv_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=0).float()
    return tpr_bool.mean(), ppv_bool.mean(), tpr_bool, ppv_bool


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
        sleep_tpr, sleep_ppv = get_summary_stats(
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
