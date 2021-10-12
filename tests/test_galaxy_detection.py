import pytest


class TestBasicGalaxyTiles:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return {
            "model": "sleep_galaxy_basic",
            "dataset": "default" if devices.use_cuda else "cpu",
            "training": "unittest" if devices.use_cuda else "cpu",
            "training.n_epochs": 201 if devices.use_cuda else 1,
        }

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, model_setup):
        return model_setup.get_trained_model(overrides)

    def test_simulated(self, overrides, trained_sleep, model_setup, devices):
        overrides.update({"testing": "default"})
        results = model_setup.test_model(overrides, trained_sleep)
        assert "avg_distance" in results

        # only check testing results if GPU available
        if not devices.use_cuda:
            return

        # check testing results are sensible.
        assert results["avg_distance"] < 1.0
