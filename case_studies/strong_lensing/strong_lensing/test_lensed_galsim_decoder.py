import pytest
from hydra.utils import instantiate


@pytest.fixture(scope="module")
def overrides(devices):
    return {}


def test_lensed_galsim_decoder(strong_lensing_setup, devices, overrides):
    cfg = strong_lensing_setup.get_cfg(overrides)
    instantiate(cfg.models.decoder)

    dataset = instantiate(
        cfg.datasets.simulated,
        generate_device="cpu",
    )

    sample_batch_size = 16
    tile_catalog = dataset.sample_prior(
        sample_batch_size, cfg.datasets.simulated.n_tiles_h, cfg.datasets.simulated.n_tiles_w
    )
    tile_catalog.set_all_fluxes_and_mags(dataset.image_decoder)

    dataset.simulate_image_from_catalog(tile_catalog)
