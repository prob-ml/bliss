import numpy as np
import pytest

from bliss.surveys.sdss import SloanDigitalSkySurvey


class TestSDSS:
    def test_sdss(self, cfg):
        sdss_dir = cfg.paths.sdss
        sdss_obj = SloanDigitalSkySurvey(
            sdss_dir,
            run=3900,
            camcol=6,
            fields=[269],
        )
        an_obj = sdss_obj[0]
        for k in ("image", "background", "gain", "nelec_per_nmgy_list", "calibration"):
            assert isinstance(an_obj[k], np.ndarray)

        assert an_obj["field"] == 269
        assert an_obj["gain"][3] == pytest.approx(4.76)
        assert isinstance(an_obj["wcs"], list)
