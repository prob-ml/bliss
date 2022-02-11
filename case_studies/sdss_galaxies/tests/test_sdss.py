import pytest

from bliss.datasets.sdss import SloanDigitalSkySurvey


class TestSDSS:
    def test_sdss(self, devices, get_config):
        cfg = get_config({}, devices)
        sdss_dir = cfg.paths.sdss
        sdss_obj = SloanDigitalSkySurvey(
            sdss_dir,
            run=3900,
            camcol=6,
            fields=[269],
            bands=range(5),
        )
        an_obj = sdss_obj[0]
        assert an_obj["gain"][3] == pytest.approx(4.76)

        sdss_obj9 = SloanDigitalSkySurvey(
            sdss_dir,
            run=3900,
            camcol=6,
            fields=[269, 745],
            bands=range(5),
        )
        sdss_obj9[0]
