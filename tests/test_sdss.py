import pytest
from bliss.datasets import sdss
import os


class TestSDSS:
    def test_sdss(self, paths):
        sdss_dir = paths["data"].joinpath("sdss")
        sdss_obj = sdss.SloanDigitalSkySurvey(
            sdss_dir, run=2583, camcol=2, field=136, bands=range(5)
        )

        assert sdss_obj[0]["gain"][3] == pytest.approx(5.195)

        cache_file = sdss_dir.joinpath("2583/2/136/cache.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
