import pytest


class TestGalaxyTiles:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = dict(
            model="basic_sleep_galaxy",
            dataset="default" if devices.use_cuda else "cpu",
            training="unittest" if devices.use_cuda else "cpu",
        )
        return overrides

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, sleep_setup):
        return sleep_setup.get_trained_sleep(overrides)

    def test_simulated(self, overrides, trained_sleep, sleep_setup, devices):
        overrides.update({"testing": "default"})
        results = sleep_setup.test_sleep(overrides, trained_sleep)

        # only check testing results if GPU available
        if not devices.use_cuda:
            return

        # check testing results are sensible.
        assert results["acc_gal_counts"] > 0.70
        assert results["locs_mse"] < 1.0

    def test_saved(self, overrides, trained_sleep, sleep_setup, devices):
        overrides.update({"testing": "galaxy_test1"})
        results = sleep_setup.test_sleep(overrides, trained_sleep)

        # only check testing results if GPU available
        if not devices.use_cuda:
            return

        # check testing results are sensible.
        assert results["acc_gal_counts"] > 0.70
        assert results["locs_mse"] < 1.0
