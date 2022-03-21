import pytest


@pytest.fixture(scope="module")
def overrides(devices):
    overrides = {
        "mode": "train",
        "training": "sdss_binary",
    }
    if devices.use_cuda:
        overrides.update({"training.n_epochs": 50})
    else:
        overrides.update(
            {
                "datasets.simulated.n_batches": 1,
                "datasets.simulated.batch_size": 2,
                "datasets.simulated.generate_device": "cpu",
                "training.n_epochs": 2,
            }
        )
    return overrides


def test_binary(sdss_galaxies_setup, devices, overrides):
    trained_binary = sdss_galaxies_setup.get_trained_model(overrides)
    results = sdss_galaxies_setup.test_model(overrides, trained_binary)

    if devices.use_cuda:
        assert results["acc"] > 0.85


def test_binary_plotting(sdss_galaxies_setup, overrides):
    overrides.update(
        {
            "datasets.simulated.batch_size": 16,
            "training.trainer.log_every_n_steps": 1,
        }
    )
    sdss_galaxies_setup.get_trained_model(overrides)
