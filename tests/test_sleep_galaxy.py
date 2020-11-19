import pytest


class TestBasicGalaxyTiles:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = dict(
            model="sleep_galaxy_basic",
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
        assert results["locs_median_mse"] < 0.55

    def test_saved(self, overrides, trained_sleep, sleep_setup, devices, paths):
        test_file = paths["data"].joinpath("galaxy_test1.pt").as_posix()
        overrides.update({"testing.file": test_file})
        results = sleep_setup.test_sleep(overrides, trained_sleep)

        # only check testing results if GPU available
        if not devices.use_cuda:
            return

        # check testing results are sensible.
        assert results["acc_gal_counts"] > 0.70
        assert results["locs_median_mse"] < 0.5


def test_three_saved_galaxies(sleep_setup, devices, paths):
    # This shows we can train on tiles and do pretty well on images with up to 3 galaxies not tiled.
    test_file = paths["data"].joinpath("0-3_galaxies.pt").as_posix()
    overrides = dict(
        model="sleep_galaxy_2_per_tile",
        dataset="default" if devices.use_cuda else "cpu",
        training="unittest" if devices.use_cuda else "cpu",
    )
    trained_sleep = sleep_setup.get_trained_sleep(overrides)

    overrides.update({"testing.file": test_file})
    results = sleep_setup.test_sleep(overrides, trained_sleep)

    # only check testing results if GPU available
    if not devices.use_cuda:
        return

    # check testing results are sensible.
    assert results["acc_gal_counts"] > 0.70
    assert results["locs_median_mse"] < 0.5
