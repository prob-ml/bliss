from shutil import copytree

import numpy as np
import pytest
from hydra.utils import instantiate


class TestSDSS:
    def test_sdss(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.surveys.sdss.fields = [{"run": 3900, "camcol": 6, "fields": [269]}]
        sdss_obj = instantiate(the_cfg.surveys.sdss)
        sdss_obj.prepare_data()
        an_obj = sdss_obj[0]
        for k in ("background", "gain", "flux_calibration_list", "calibration"):
            assert isinstance(an_obj[k], np.ndarray)

        assert an_obj["field"] == 269
        assert an_obj["gain"][3] == pytest.approx(4.76)
        assert isinstance(an_obj["wcs"], list)

    def test_sdss_custom_dir(self, cfg, tmpdir_factory):
        the_cfg = cfg.copy()
        the_cfg.paths.root = str(tmpdir_factory.mktemp("root"))
        if cfg.surveys.sdss.dir_path != the_cfg.surveys.sdss.dir_path:
            copytree(cfg.surveys.sdss.dir_path, the_cfg.surveys.sdss.dir_path, dirs_exist_ok=True)
        # Also tests images loaded
        sdss_obj = instantiate(the_cfg.surveys.sdss, load_image_data=True)
        sdss_obj.prepare_data()
        frame0 = sdss_obj[0]
        assert frame0["image"].shape == (5, 1488, 2048)
