import pytest
from bliss.datasets import sdss
import os
import numpy as np


class TestSDSS:
    def test_sdss(self, paths):
        sdss_dir = paths["data"].joinpath("sdss")
        sdss_obj = sdss.SloanDigitalSkySurvey(
            sdss_dir, run=3900, camcol=6, fields=[269], bands=range(5)
        )
        assert sdss_obj[0]["gain"][3] == pytest.approx(4.76)

        assert len(sdss_obj[0]["bright_stars"]) == 43
        super_star = sdss_obj[0]["bright_stars"].sum(axis=0)
        assert np.all(super_star[2, 2] + 1e-4 >= super_star)

        cache_file = sdss_dir.joinpath("3900/6/269/cache.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
