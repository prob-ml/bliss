import pytest


class TestBasicGalaxyTiles:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = {
            "model": "sleep_galaxy_basic",
            "dataset": "default" if devices.use_cuda else "cpu",
            "training": "unittest" if devices.use_cuda else "cpu",
            "training.n_epochs": 201 if devices.use_cuda else 1,
        }
        return overrides

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, sleep_setup):
        return sleep_setup.get_trained_sleep(overrides)

    def test_simulated(self, overrides, trained_sleep, sleep_setup, devices):
        overrides.update({"testing": "default"})
        results = sleep_setup.test_sleep(overrides, trained_sleep)
        assert {"acc_gal_counts", "locs_mae"}.issubset(results.keys())

        # only check testing results if GPU available
        if not devices.use_cuda:
            return

        # check testing results are sensible.
        assert results["acc_gal_counts"] > 0.7
        assert results["locs_mae"] < 0.85
