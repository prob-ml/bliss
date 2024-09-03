from shutil import copytree

import numpy as np
import pytest
from hydra.utils import instantiate


@pytest.fixture(autouse=True)
def patch_align(monkeypatch):
    # align is quite slow, so we replace it with the identity function
    identity = lambda x, *_args, **_kwargs: x
    monkeypatch.setattr("bliss.surveys.survey.align", identity)


class TestSDSS:
    def test_sdss(self, cfg):
        cfg.surveys.sdss.fields = [{"run": 3900, "camcol": 6, "fields": [269]}]
        sdss_obj = instantiate(cfg.surveys.sdss)
        sdss_obj.prepare_data()
        an_obj = sdss_obj[0]
        for k in ("background", "gain", "flux_calibration"):
            assert isinstance(an_obj[k], np.ndarray)

        assert an_obj["field"] == 269
        assert an_obj["gain"][3] == pytest.approx(4.76)
        assert isinstance(an_obj["wcs"], list)

    def test_sdss_custom_dir(self, cfg):
        if cfg.surveys.sdss.dir_path != cfg.surveys.sdss.dir_path:
            copytree(cfg.surveys.sdss.dir_path, cfg.surveys.sdss.dir_path, dirs_exist_ok=True)
        # Also tests images loaded
        sdss_obj = instantiate(cfg.surveys.sdss, load_image_data=True)
        sdss_obj.prepare_data()
        frame0 = sdss_obj[0]
        assert frame0["image"].shape == (5, 1489, 2048)
