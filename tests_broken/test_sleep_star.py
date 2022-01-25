import pytest


class TestSleepStarOneTile:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return {
            "model": "sleep_star_one_tile",
            "dataset": ("single_tile" if devices.use_cuda else "cpu"),
            "training": ("unittest" if devices.use_cuda else "cpu"),
            "optimizer": "m2",
        }

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, model_setup):
        return model_setup.get_trained_model(overrides)

    def test_simulated(self, overrides, trained_sleep, model_setup, devices):
        overrides.update({"testing": "default"})
        results = model_setup.test_model(overrides, trained_sleep)
        assert {"precision", "f1", "avg_distance"}.issubset(results.keys())

        # only expect tests to pass if gpu
        if not devices.use_cuda:
            return

        assert results["precision"] > 0.85
        assert results["f1"] > 0.8
        assert results["avg_distance"] < 1.0


class TestSleepStarTiles:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return {
            "model": "sleep_star_basic",
            "dataset": "default" if devices.use_cuda else "cpu",
            "training": "unittest" if devices.use_cuda else "cpu",
            "optimizer": "m2",
        }

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, model_setup):
        return model_setup.get_trained_model(overrides)

    def test_simulated(self, overrides, trained_sleep, model_setup, devices):
        overrides.update({"testing": "default"})
        results = model_setup.test_model(overrides, trained_sleep)
        assert {"precision", "f1", "avg_distance"}.issubset(results.keys())

        # only expect tests to pass if gpu
        if not devices.use_cuda:
            return

        assert results["precision"] > 0.85
        assert results["f1"] > 0.8
        assert results["avg_distance"] < 1.0
