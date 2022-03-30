from pathlib import Path

import galsim
import torch
from astropy.table import Table
from hydra.utils import instantiate

from bliss import reporting
from bliss.catalog import FullCatalog


def test_scene_metrics():
    true_params = FullCatalog(
        100,
        100,
        {
            "plocs": torch.tensor([[[50.0, 50.0]]]).float(),
            "galaxy_bools": torch.tensor([[[1]]]).bool(),
            "n_sources": torch.tensor([1]).long(),
            "mags": torch.tensor([[[23.0]]]).float(),
        },
    )
    reporting.scene_metrics(true_params, true_params, mag_max=25, slack=1.0, mag_slack=0.25)


def test_catalog_conversion():
    true_params = FullCatalog(
        100,
        100,
        {
            "plocs": torch.tensor([[[51.8, 49.6]]]).float(),
            "galaxy_bools": torch.tensor([[[1]]]).bool(),
            "n_sources": torch.tensor([1]).long(),
            "mags": torch.tensor([[[23.0]]]).float(),
        },
    )
    true_tile_params = true_params.to_tile_params(tile_slen=4, max_sources_per_tile=1)
    tile_params_tilde = true_tile_params.to_full_params()
    assert true_params.equals(tile_params_tilde)


def test_coadd(devices, get_sdss_galaxies_config):

    # get psf
    cfg = get_sdss_galaxies_config({}, devices)
    ds = instantiate(cfg.datasets.sdss_galaxies)
    psf = ds.psf

    # read file and get flux / hlr
    coadd_cat_file = Path(cfg.paths.data).joinpath("coadd_catalog_94_1_12.fits")
    coadd_cat = Table.read(coadd_cat_file)
    coadd_cat.remove_column("hlr")
    reporting.get_flux_coadd(coadd_cat[:5])
    reporting.get_hlr_coadd(coadd_cat[:5], psf)
    coadd_cat["hlr"] = 0.0

    # params for calculating metrics
    h = 1489
    w = 2048
    bp = 24
    reporting.CoaddFullCatalog.from_table(coadd_cat, (bp, h - bp), (bp, w - bp))


def test_measurements():
    slen = 40
    pixel_scale = 0.2
    psf = galsim.Gaussian(sigma=0.2)
    gal = galsim.Gaussian(sigma=1.0)
    gal_conv = galsim.Convolution(gal, psf)
    psf_image = psf.drawImage(nx=slen, ny=slen, scale=pixel_scale).array.reshape(1, slen, slen)
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale).array.reshape(1, 1, slen, slen)

    reporting.get_single_galaxy_measurements(slen, image, image, psf_image, pixel_scale)
