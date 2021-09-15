import numpy as np
import pytest

from bliss.datasets import sdss


class TestSDSS:
    def test_sdss(self, paths):
        sdss_dir = paths["sdss"]
        sdss_obj = sdss.SloanDigitalSkySurvey(
            sdss_dir,
            run=3900,
            camcol=6,
            fields=[269],
            bands=range(5),
            stampsize=5,
            overwrite_cache=True,
            overwrite_fits_cache=True,
        )
        an_obj = sdss_obj[0]
        assert an_obj["gain"][3] == pytest.approx(4.76)

        assert len(an_obj["bright_stars"]) == 43
        assert an_obj["bright_stars"].shape[1] == 5
        assert an_obj["bright_stars"].shape[2] == 5

        assert len(an_obj["sdss_psfs"]) == 43
        assert an_obj["sdss_psfs"].shape[1] == 5
        assert an_obj["sdss_psfs"].shape[2] == 5
        super_star = an_obj["bright_stars"].sum(axis=0)
        assert np.all(super_star[2, 2] + 1e-4 >= super_star)

        assert len(an_obj["sdss_psfs"]) == 43

        sdss_obj9 = sdss.SloanDigitalSkySurvey(
            sdss_dir, run=3900, camcol=6, fields=[269, 745], bands=range(5), stampsize=9
        )

        another_obj = sdss_obj9[0]
        assert another_obj["bright_stars"].shape[1] == 9
        assert another_obj["bright_stars"].shape[2] == 9
        assert another_obj["sdss_psfs"].shape[1] == 9
        assert another_obj["sdss_psfs"].shape[2] == 9

        sdss_obj9_cached = sdss.SloanDigitalSkySurvey(
            sdss_dir, run=3900, camcol=6, fields=[269, 745], bands=range(5), stampsize=9
        )

        assert sdss_obj9_cached[0]["bright_stars"].shape[1] == 9
        assert sdss_obj9_cached[0]["bright_stars"].shape[2] == 9

        sdss_obj.clear_cache()
        sdss_obj9.clear_cache()
        sdss_obj9_cached.clear_cache()
