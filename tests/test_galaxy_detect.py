import pytest


class TestBasicGalaxyTiles:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = dict(
            model="sleep_galaxy_detection_basic",
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
        assert {"acc_gal_counts", "locs_mae"}.issubset(results.keys())

        # only check testing results if GPU available
        if not devices.use_cuda:
            return

        # check testing results are sensible.
        assert results["acc_gal_counts"] > 0.70
        assert results["locs_mae"] < 0.65

    def test_saved(self, overrides, trained_sleep, sleep_setup, devices, paths):
        test_file = paths["data"].joinpath("galaxy_test1.pt").as_posix()
        overrides.update({"testing.file": test_file})
        results = sleep_setup.test_sleep(overrides, trained_sleep)

        # only check testing results if GPU available
        if not devices.use_cuda:
            return

        # check testing results are sensible.
        assert results["acc_gal_counts"] > 0.70
        assert results["locs_mae"] < 0.60
