import numpy as np
import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler

from celeste import use_cuda, psf_transform, wake, sleep
from celeste.models import decoder, encoder


@pytest.fixture(scope="module")
def init_psf_setup(data_path, device):
    psf_file = data_path.joinpath("fitted_powerlaw_psf_params.npy")
    true_psf_params = torch.from_numpy(np.load(psf_file)).to(device)
    init_psf_params = true_psf_params.clone()[None, 0, ...]
    init_psf_params[0, 1:3] += torch.tensor([1.0, 1.0]).to(device)

    init_psf = psf_transform.PowerLawPSF(init_psf_params).forward().detach()

    return {"init_psf_params": init_psf_params, "init_psf": init_psf}


def test_benchmark(
    benchmark,
    test_3_stars,
    single_band_galaxy_decoder,
    init_psf_setup,
    device,
    device_id,
    sprof,
    wprof,
    log,
    n_bands=1,
    max_stars=20,
    mean_stars=15,
    min_stars=5,
    f_min=1e4,
    slen=50,
    n_images=128,
    batch_size=32,
    n_epochs=200,
    prob_galaxy=0.0,
):
    # load the test image
    # 3-stars 30*30
    test_image = test_3_stars["images"]

    # initialization
    # initialize background params, which will create the true background
    background = torch.zeros(n_bands, slen, slen, device=device)
    background.fill_(686.0)

    init_background_params = torch.zeros(1, 3, device=device)
    init_background_params[0, 0] = 686.0

    # initialize psf params, just add 4 to each sigmas
    init_psf_params = init_psf_setup["init_psf"]

    n_samples = 1000 if use_cuda else 1
    hparams = {"n_samples": n_samples, "lr": 0.001}

    simulator_args = (
        single_band_galaxy_decoder,
        init_psf_params,
        background,
    )

    simulator_kwargs = dict(
        slen=slen,
        n_bands=n_bands,
        max_sources=max_stars,
        mean_sources=mean_stars,
        min_sources=min_stars,
        f_min=f_min,
        prob_galaxy=prob_galaxy,
    )

    n_batches = int(n_images / batch_size)
    dataset = decoder.SimulatedDataset(
        n_batches, batch_size, simulator_args, simulator_kwargs
    )

    image_encoder = encoder.ImageEncoder(
        slen=slen,
        ptile_slen=8,
        step=2,
        edge_padding=3,
        n_bands=n_bands,
        max_detections=2,
        n_galaxy_params=single_band_galaxy_decoder.latent_dim,
        enc_conv_c=5,
        enc_kern=3,
        enc_hidden=64,
    ).to(device)

    sleep_net = sleep.SleepPhase(
        dataset=dataset, image_encoder=image_encoder, save_logs=log
    )

    profiler = AdvancedProfiler(output_filename=sprof) if sprof != None else None

    # runs on gpu or cpu?
    n_device = [device_id] if use_cuda else 0  # 0 means no gpu

    # initiate trainers
    sleep_trainer = pl.Trainer(
        logger=log,
        checkpoint_callback=log,
        gpus=n_device,
        profiler=profiler,
        min_epochs=n_epochs,
        max_epochs=n_epochs,
        reload_dataloaders_every_epoch=True,
        max_steps=1,
    )

    # wake_phase_model = wake.WakePhase(
    #    image_encoder,
    #    test_image,
    #    init_psf_params,
    #    init_background_params,
    #    hparams,
    #    log,
    # )

    with torch.no_grad():
        benchmark.pedantic(
            sleep_trainer.fit, args=(sleep_net,), rounds=10, iterations=5
        )
