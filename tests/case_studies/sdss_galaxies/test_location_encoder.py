import pytest


@pytest.fixture(scope="module")
def overrides(devices):
    overrides = {
        "mode": "train",
        "training": "sdss_sleep",
    }
    if devices.use_cuda:
        overrides.update({"training.n_epochs": 201})
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


def test_sdss_location_encoder(sdss_galaxies_setup, overrides, devices):
    trained_location = sdss_galaxies_setup.get_trained_model(overrides)
    results = sdss_galaxies_setup.test_model(overrides, trained_location)

    assert "avg_distance" in results
    assert "precision" in results
    assert "f1" in results

    # only check testing results if GPU available
    if not devices.use_cuda:
        return

    # check testing results are sensible.
    assert results["avg_distance"] < 1.5
    assert results["precision"] > 0.8
    assert results["f1"] > 0.8


def test_location_encoder_plotting(sdss_galaxies_setup, overrides):
    # just to test `make_validation_plots` works.
    overrides.update(
        {
            "datasets.simulated.batch_size": 16,
            "models.sleep.annotate_probs": True,
            "training.trainer.log_every_n_steps": 1,
        }
    )
    sdss_galaxies_setup.get_trained_model(overrides)
