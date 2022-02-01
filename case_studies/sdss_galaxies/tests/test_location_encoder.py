import pytest

from bliss.train import train


@pytest.fixture(scope="module")
def overrides(devices):
    overrides = {
        "mode": "train",
        "training": "sdss_sleep",
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


def test_location_encoder(overrides, devices, get_config):
    cfg = get_config(overrides, devices)
    train(cfg)


def test_location_encoder_plotting(overrides, model_setup):
    # just to test `make_validation_plots` works.
    overrides.update(
        {
            "datasets.simulated.batch_size": 16,
            "models.sleep.annotate_probs": True,
            "training.trainer.log_every_n_steps": 1,
        }
    )
    model_setup.get_trained_model(overrides)
