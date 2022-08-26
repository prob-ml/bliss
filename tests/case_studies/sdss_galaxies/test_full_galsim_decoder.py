import pytest


@pytest.fixture(scope="module")
def overrides():
    return {
        "mode": "train",
        "training": "sdss_detection_encoder_full_decoder",
        "training.n_epochs": 11,
        "training.trainer.check_val_every_n_epoch": 2,
        "datasets.galsim_blends.num_workers": 0,
        "datasets.galsim_blends.batch_size": 1,
        "datasets.galsim_blends.n_batches": 1,
        "datasets.galsim_blends.fix_validation_set": False,
        "datasets.galsim_blends.prior.galaxy_prob": 0.5,
    }


def test_sdss_detection_encoder(sdss_galaxies_setup, overrides):
    sdss_galaxies_setup.get_trained_model(overrides)
