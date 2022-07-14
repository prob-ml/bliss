import pytest


@pytest.fixture(scope="module")
def overrides(devices):
    overrides = {
        "mode": "train",
        "training": "sdss_lensing_binary_encoder",
    }
    if devices.use_cuda:
        overrides.update({"training.n_epochs": 51})
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


def test_sdss_lensing_detection_encoder(strong_lensing_setup, overrides, devices):
    trained_location = strong_lensing_setup.get_trained_model(overrides)
    results = strong_lensing_setup.test_model(overrides, trained_location)

    assert "acc" in results

    # only check testing results if GPU available
    if devices.use_cuda:
        assert results["acc"] > 0.75
