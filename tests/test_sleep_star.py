import pytest


class TestSleepStarOneTile:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = dict(
            model="basic_sleep_star_one_tile",
            dataset="single_tile" if devices.use_cuda else "cpu",
            training="unittest" if devices.use_cuda else "cpu",
        )
        return overrides

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, sleep_setup):
        return sleep_setup.get_trained_sleep(overrides)

    def test_simulated(self, overrides, trained_sleep, sleep_setup, devices):
        overrides.update({"testing": "default"})
        results = sleep_setup.test_sleep(overrides, trained_sleep)

        # only expect tests to pass if gpu
        if not devices.use_cuda:
            return

        assert results["acc_counts"] > 0.7
        assert results["locs_mse"] < 1.0


class TestSleepStarTiles:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = dict(
            model="basic_sleep_star",
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

        # only expect tests to pass if gpu
        if not devices.use_cuda:
            return

        assert results["acc_counts"] > 0.7
        assert results["locs_mse"] < 1.0

    def test_saved(self, overrides, trained_sleep, sleep_setup, devices):
        overrides.update({"testing": "star_test1"})
        results = sleep_setup.test_sleep(overrides, trained_sleep)

        # only expect tests to pass if gpu
        if not devices.use_cuda:
            return

        assert results["acc_counts"] > 0.7
        assert results["locs_mse"] < 1.0
