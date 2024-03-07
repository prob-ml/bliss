import pytest


@pytest.fixture(scope="module")
def overrides(devices):
    overrides = {
        "mode": "train",
        "training": "binary_encoder",
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


def test_binary(galsim_galaxies_setup, devices, overrides):
    trained_binary = galsim_galaxies_setup.get_trained_model(overrides)
    results = galsim_galaxies_setup.test_model(overrides, trained_binary)

    if devices.use_cuda:
        assert results["acc"] > 0.75


def test_binary_plotting(galsim_galaxies_setup, overrides):
    overrides.update(
        {
            "datasets.simulated.batch_size": 16,
            "training.trainer.log_every_n_steps": 1,
        }
    )
    galsim_galaxies_setup.get_trained_model(overrides)
