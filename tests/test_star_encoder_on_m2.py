import torch
import pytest
import os

import pytorch_lightning as pl

import numpy as np

from bliss import use_cuda, sleep
from bliss.models import decoder, encoder
from bliss.datasets import simulated

from bliss import psf_transform
from bliss import image_statistics

torch.manual_seed(84)
np.random.seed(43)


@pytest.fixture(scope="module")
def trained_star_encoder_m2(
    data_path, device, device_id, profiler, save_logs, logs_path
):

    # dataset parameters
    n_bands = 2
    max_stars = 2500
    mean_stars = 1200
    min_stars = 0
    f_min = 1e3
    f_max = 1000000
    alpha = 0.5
    slen = 100

    # set background
    background = torch.zeros(n_bands, slen, slen, device=device)
    background[0] = 686.0
    background[1] = 1123.0

    # load SDSS PSF
    psf_file = os.path.join(data_path, "psField-002583-2-0136.fits")
    init_psf_params = psf_transform.get_psf_params(psf_file, bands=[2, 3])
    power_law_psf = psf_transform.PowerLawPSF(init_psf_params.to(device))
    psf_og = power_law_psf.forward().detach()

    # create simulated dataset
    # simulate dataset
    n_images = 200 if use_cuda else 4
    simulator_args = (
        None,
        psf_og,
        background,
    )

    simulator_kwargs = dict(
        slen=slen,
        n_bands=n_bands,
        prob_galaxy=0.0,
        max_sources=max_stars,
        mean_sources=mean_stars,
        min_sources=min_stars,
        f_min=f_min,
        f_max=f_max,
        alpha=alpha,
    )

    batch_size = 20 if use_cuda else 4
    n_batches = int(n_images / batch_size)
    dataset = decoder.SimulatedDataset(
        n_batches, batch_size, simulator_args, simulator_kwargs
    )

    # set up star encoder
    star_encoder = encoder.ImageEncoder(
        slen=slen,
        ptile_slen=8,
        step=2,
        edge_padding=3,
        n_bands=n_bands,
        max_detections=2,
        n_galaxy_params=dataset.simulator.latent_dim,
        enc_conv_c=20,
        enc_kern=3,
        enc_hidden=256,
    ).to(device)

    # set up training module
    # profiler = AdvancedProfiler(output_filename=sprof)
    sleep_net = sleep.SleepPhase(
        dataset=dataset, image_encoder=star_encoder, save_logs=save_logs
    )
    n_device = [device_id] if use_cuda else 0
    n_epochs = 200 if use_cuda else 1
    sleep_trainer = pl.Trainer(
        gpus=n_device,
        profiler=profiler,
        min_epochs=n_epochs,
        max_epochs=n_epochs,
        reload_dataloaders_every_epoch=True,
        default_root_dir=logs_path,
    )

    sleep_trainer.fit(sleep_net)

    return sleep_net.image_encoder


class TestStarSleepEncoderM2:
    def test_star_sleep_m2(self, data_path, trained_star_encoder_m2):

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

        # check metrics if cuda is true
        if not use_cuda:
            return

        assert sleep_tpr > 0.45
        assert sleep_ppv > 0.35
