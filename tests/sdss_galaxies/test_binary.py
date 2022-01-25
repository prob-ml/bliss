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


def test_binary(model_setup, devices, overrides):
    trained_binary = model_setup.get_trained_model(overrides)
    results = model_setup.test_model(overrides, trained_binary)

    if devices.use_cuda:
        assert results["acc"] > 0.85


def test_binary_plotting(model_setup, overrides):
    overrides.update(
        {
            "datasets.simulated.batch_size": 16,
            "training.trainer.log_every_n_steps": 1,
        }
    )
    model_setup.get_trained_model(overrides)
