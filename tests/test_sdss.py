from shutil import copytree

import numpy as np
import pytest
from hydra.utils import instantiate


class TestSDSS:
    def test_sdss(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.surveys.sdss.fields = [{"run": 3900, "camcol": 6, "fields": [269]}]
        sdss_obj = instantiate(the_cfg.surveys.sdss)
        an_obj = sdss_obj[0]
        for k in ("image", "background", "gain", "nelec_per_nmgy_list", "calibration"):
            assert isinstance(an_obj[k], np.ndarray)

        assert an_obj["field"] == 269
        assert an_obj["gain"][3] == pytest.approx(4.76)
        assert isinstance(an_obj["wcs"], list)

    def test_sdss_custom_dir(self, cfg, tmpdir_factory):
        the_cfg = cfg.copy()
        the_cfg.paths.root = str(tmpdir_factory.mktemp("root"))
        copytree(
            cfg.surveys.sdss.dir_path + "/color_models",
            the_cfg.surveys.sdss.dir_path + "/color_models",
        )
        sdss_obj = instantiate(the_cfg.surveys.sdss)[0]
        assert sdss_obj["image"].shape == (5, 1489, 2048)
