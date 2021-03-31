import pytest
from .test_sleep_star import TestSleepStarOneTile


class TestSleepStarOneTileHNP(TestSleepStarOneTile):
    @pytest.fixture(scope="class")
    def overrides(self, devices):
        overrides = dict(
            model="sleep_star_one_tile_hnp",
            dataset="single_tile" if devices.use_cuda else "cpu",
            training="unittest" if devices.use_cuda else "cpu",
            optimizer="m2",
        )
        return overrides
