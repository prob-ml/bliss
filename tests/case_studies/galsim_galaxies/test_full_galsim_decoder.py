import pytest
from hydra.utils import instantiate


@pytest.fixture(scope="module")
def overrides():
    return {
        "datasets.galsim_blends.num_workers": 0,
        "datasets.galsim_blends.batch_size": 1,
        "datasets.galsim_blends.n_batches": 1,
        "datasets.galsim_blends.prior.galaxy_prob": 0.5,
    }


def test_sdss_detection_encoder(galsim_galaxies_setup, overrides):
    cfg = galsim_galaxies_setup.get_cfg(overrides)
    ds = instantiate(cfg.datasets.galsim_blends)
    assert ds[0] is not None
