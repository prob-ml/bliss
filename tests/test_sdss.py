from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from hydra import compose, initialize
from hydra.utils import instantiate

from bliss.datasets.sdss import SloanDigitalSkySurvey, get_flux_coadd, get_hlr_coadd


class TestSDSS:
    def test_sdss(self, paths):
        sdss_dir = paths["sdss"]
        sdss_obj = SloanDigitalSkySurvey(
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

        sdss_obj9 = SloanDigitalSkySurvey(
            sdss_dir, run=3900, camcol=6, fields=[269, 745], bands=range(5), stampsize=9
        )

        another_obj = sdss_obj9[0]
        assert another_obj["bright_stars"].shape[1] == 9
        assert another_obj["bright_stars"].shape[2] == 9
        assert another_obj["sdss_psfs"].shape[1] == 9
        assert another_obj["sdss_psfs"].shape[2] == 9

        sdss_obj9_cached = SloanDigitalSkySurvey(
            sdss_dir, run=3900, camcol=6, fields=[269, 745], bands=range(5), stampsize=9
        )

        assert sdss_obj9_cached[0]["bright_stars"].shape[1] == 9
        assert sdss_obj9_cached[0]["bright_stars"].shape[2] == 9

        sdss_obj.clear_cache()
        sdss_obj9.clear_cache()
        sdss_obj9_cached.clear_cache()

    def test_coadd(self, paths):

        # get psf
        with initialize(config_path="../config"):
            cfg = compose("config", overrides=["dataset=sdss_galaxies"])
        ds = instantiate(cfg.dataset)
        psf = ds.psf
        coadd_cat_file = Path(paths["data"]).joinpath("coadd_catalog_94_1_12.fits")
        coadd_cat = Table.read(coadd_cat_file)
        _ = get_flux_coadd(coadd_cat)
        _ = get_hlr_coadd(coadd_cat[:5], psf)
