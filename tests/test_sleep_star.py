from pathlib import Path

import pytest


class TestSleepStarOneTile:
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        return dict(
            model="sleep_star_one_tile",
            dataset="single_tile" if devices.use_cuda else "cpu",
            training="unittest" if devices.use_cuda else "cpu",
            optimizer="m2",
        )

    @pytest.fixture(scope="class")
    def trained_sleep(self, overrides, model_setup):
        return model_setup.get_trained_model(overrides)

    def test_simulated(self, overrides, trained_sleep, model_setup, devices):
        overrides.update({"testing": "default"})
        results = model_setup.test_model(overrides, trained_sleep)
        assert {"acc_counts", "locs_mae", "star_fluxes_mae"}.issubset(results.keys())

        # only expect tests to pass if gpu
        if not devices.use_cuda:
            return

        assert results["acc_counts"] > 0.9
        assert results["locs_mae"] < 0.5
        assert results["star_fluxes_mae"] < 0.5


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
        assert {"acc_counts", "locs_mae", "star_fluxes_mae"}.issubset(results.keys())

        # only expect tests to pass if gpu
        if not devices.use_cuda:
            return

        assert results["acc_counts"] > 0.7
        assert results["locs_mae"] < 0.5
        assert results["star_fluxes_mae"] < 0.5

    def test_saved(self, overrides, trained_sleep, model_setup, devices, paths):
        test_file = Path(paths["data"]).joinpath("star_test1.pt").as_posix()
        overrides.update({"testing.file": test_file})
        results = model_setup.test_model(overrides, trained_sleep)

        # only expect tests to pass if gpu
        if not devices.use_cuda:
            return

        assert results["acc_counts"] > 0.7
        assert results["locs_mae"] < 0.5
        assert results["star_fluxes_mae"] < 0.5
