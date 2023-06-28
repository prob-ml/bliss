import numpy as np
import pytest
from mock_tests import MockSDSSDownloader, mock_get, mock_post

from bliss.surveys import sdss
from bliss.surveys.sdss import SloanDigitalSkySurvey


class TestSDSS:
    def test_sdss(self, monkeypatch, tmpdir):
        monkeypatch.setattr("requests.get", mock_get)
        monkeypatch.setattr("requests.post", mock_post)
        monkeypatch.setattr("gzip.decompress", lambda x: x)
        monkeypatch.setattr("bz2.decompress", lambda x: x)
        monkeypatch.setattr(sdss, "SDSSDownloader", MockSDSSDownloader)

        sdss_dir = f"{tmpdir}/data/sdss"
        sdss_obj = SloanDigitalSkySurvey(
            sdss_dir,
            run=3900,
            camcol=6,
            fields=[269],
            bands=range(5),
        )
        an_obj = sdss_obj[0]
        for k in ("image", "background", "gain", "nelec_per_nmgy_list", "calibration"):
            assert isinstance(an_obj[k], np.ndarray)

        assert an_obj["field"] == 269
        assert an_obj["gain"][3] == pytest.approx(4.76)
        assert isinstance(an_obj["wcs"], list)

        # fake download 745 and just check that the size is right
        sdss_obj9 = SloanDigitalSkySurvey(
            sdss_dir,
            run=3900,
            camcol=6,
            fields=[269, 745],
            bands=range(5),
        )
        assert (len(sdss_obj9)) == 2
