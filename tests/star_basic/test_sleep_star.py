import pytest

from bliss.train import train


# TODO: add some tests of blended stars and galaxies
# TODO: add a test with one tile
class TestStarBasic:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return {
            "training.n_epochs": 201 if devices.use_cuda else 2,
            "simulator.generate_device": "cuda:0" if devices.use_cuda else "cpu",
        }

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, star_basic_model_setup):
        return star_basic_model_setup.get_trained_model(overrides)

    def test_simulated(self, overrides, trained_sleep, star_basic_model_setup, devices):
        results = star_basic_model_setup.test_model(overrides, trained_sleep)
        assert {"precision", "f1", "avg_distance"}.issubset(results.keys())

        # we only expect the tests below to pass if we're training on the gpu
        if not devices.use_cuda:
            return

        assert results["precision"] > 0.85
        assert results["f1"] > 0.8
        assert results["avg_distance"] < 1.0

    def test_train(self, cfg):
        train(cfg)
