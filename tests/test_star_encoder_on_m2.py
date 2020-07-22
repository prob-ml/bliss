import torch
import pytest
import os
import numpy as np

from bliss import use_cuda, image_statistics
from bliss.models import decoder

torch.manual_seed(84)
np.random.seed(43)


@pytest.fixture(scope="module")
def trained_star_encoder_m2(data_path, device, get_star_dataset, get_trained_encoder):

    # load SDSS PSF
    psf_file = os.path.join(data_path, "psField-002583-2-0136.fits")
    init_psf_params = decoder.get_psf_params(psf_file, bands=[2, 3]).to(device)
    n_epochs = 200 if use_cuda else 1

    star_dataset = get_star_dataset(
        init_psf_params,
        n_bands=2,
        slen=100,
        n_images=200 if use_cuda else 5,
        batch_size=20 if use_cuda else 5,
        max_sources=2500,
        mean_sources=1200,
        min_sources=0,
        f_min=1e3,
        f_max=1e6,
        alpha=0.5,
    )
    trained_encoder = get_trained_encoder(
        star_dataset, n_epochs=n_epochs, enc_conv_c=20, enc_kern=3, enc_hidden=256,
    )
    return trained_encoder


class TestStarSleepEncoderM2:
    def test_star_sleep_m2(self, data_path, device, trained_star_encoder_m2):

        # the trained star encoder
        trained_star_encoder_m2.eval()

        # load hubble parameters and SDSS image
        hubble_data = np.load(os.path.join(data_path, "true_hubble_m2.npy"))

        # the SDSS image
        test_image = torch.from_numpy(hubble_data["sdss_image"]).unsqueeze(0).to(device)

        # the true parameters
        true_locs = torch.from_numpy(hubble_data["true_locs"]).to(device)
        true_fluxes = torch.from_numpy(hubble_data["true_fluxes"]).to(device)
        nelec_per_nmgy = torch.from_numpy(hubble_data["nelec_per_nmgy"]).to(device)

        # get estimated parameters
        (
            n_sources,
            est_locs,
            galaxy_params,
            est_log_fluxes,
            galaxy_bool,
        ) = trained_star_encoder_m2.sample_encoder(
            test_image.to(device),
            n_samples=1,
            return_map_n_sources=True,
            return_map_source_params=True,
        )
        est_fluxes = est_log_fluxes.exp()

        # check metrics if cuda is true
        if not use_cuda:
            return

        # summary statistics
        sleep_tpr, sleep_ppv = image_statistics.get_summary_stats(
            est_locs[0],
            true_locs,
            trained_star_encoder_m2.slen,
            est_fluxes[0, :, 0],
            true_fluxes[:, 0],
            nelec_per_nmgy,
        )[0:2]

        print("Sleep phase TPR: ", sleep_tpr)
        print("Sleep phase PPV: ", sleep_ppv)

        assert sleep_tpr > 0.45
        assert sleep_ppv > 0.33
