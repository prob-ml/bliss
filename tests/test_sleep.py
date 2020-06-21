import torch
import pytest

from celeste import use_cuda
from celeste import train
from celeste.models import encoder, decoder


@pytest.fixture(scope="module")
def trained_encoder(
    data_path, single_band_galaxy_decoder, single_band_fitted_powerlaw_psf, device,
):

    # create training dataset
    n_bands = 1
    max_stars = 20
    mean_stars = 15
    min_stars = 5
    f_min = 1e4
    slen = 50

    # set background
    background = torch.zeros(n_bands, slen, slen, device=device)
    background[0] = 686.0

    # simulate dataset
    n_images = 128
    simulator_args = (
        single_band_galaxy_decoder,
        single_band_fitted_powerlaw_psf,
        background,
    )

    simulator_kwargs = dict(
        slen=slen,
        n_bands=n_bands,
        max_sources=max_stars,
        mean_sources=mean_stars,
        min_sources=min_stars,
        f_min=f_min,
        prob_galaxy=0.0,  # enforce only stars are created in the training images.
    )

    dataset = decoder.SourceDataset(n_images, simulator_args, simulator_kwargs)

    # setup Star Encoder
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

    # train encoder
    # training wrapper
    SleepTraining = train.SleepTraining(
        model=image_encoder,
        dataset=dataset,
        slen=slen,
        n_bands=n_bands,
        verbose=False,
        batchsize=32,
    )

    n_epochs = 150 if use_cuda else 1
    SleepTraining.run(n_epochs=n_epochs)

    return image_encoder


class TestStarSleepEncoder:
    def test_star_sleep(self, trained_encoder, test_star, device):

        # load test image
        test_image = test_star["images"]

        with torch.no_grad():
            # get the estimated params
            trained_encoder.eval()
            (
                n_sources,
                locs,
                galaxy_params,
                log_fluxes,
                galaxy_bool,
            ) = trained_encoder.sample_encoder(
                test_image.to(device),
                n_samples=1,
                return_map_n_sources=True,
                return_map_source_params=True,
            )

        # we only expect our assert statements to be true
        # when the model is trained in full, which requires cuda
        if not use_cuda:
            return

        # test that parameters match.
        assert n_sources == test_star["n_sources"].to(device)

        diff_locs = test_star["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 0.5

        # fluxes
        diff = test_star["log_fluxes"].sort(1)[0].to(device) - log_fluxes.sort(1)[0]
        assert torch.all(diff.abs() <= log_fluxes.sort(1)[0].abs() * 0.10)
        assert torch.all(
            diff.abs() <= test_star["log_fluxes"].sort(1)[0].abs().to(device) * 0.10
        )
