import pytest


class TestSleepStarOneTile:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return {
            "models.prior": "${priors.star_single_tile}",
            "models.decoder": "${decoders.single_tile}",
            "training.dataset": "${datasets.simulated_single_tile}",
            "training.n_epochs": 201 if devices.use_cuda else 2,
        }

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, star_basic_model_setup):
        return star_basic_model_setup.get_trained_model(overrides)

    def test_simulated(self, overrides, trained_sleep, star_basic_model_setup, devices):
        results = star_basic_model_setup.test_model(overrides, trained_sleep)
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
            "training.n_epochs": 201 if devices.use_cuda else 2,
        }

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, star_basic_model_setup):
        return star_basic_model_setup.get_trained_model(overrides)

    def test_simulated(self, overrides, trained_sleep, star_basic_model_setup, devices):
        results = star_basic_model_setup.test_model(overrides, trained_sleep)
        assert {"precision", "f1", "avg_distance"}.issubset(results.keys())

        # only expect tests to pass if gpu
        if not devices.use_cuda:
            return

        assert results["precision"] > 0.85
        assert results["f1"] > 0.8
        assert results["avg_distance"] < 1.0
