import pytest


@pytest.fixture(scope="module")
def overrides(devices):
    overrides = {
        "mode": "train",
        "training": "lens_encoder",
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


def test_lens_encoder(strong_lensing_setup, devices, overrides):
    trained_lens_encoder = strong_lensing_setup.get_trained_model(overrides)
    strong_lensing_setup.test_model(overrides, trained_lens_encoder)
