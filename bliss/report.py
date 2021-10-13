"""Functions to generate metrics on real data."""
import torch
from astropy.table import Table
from astropy.wcs.wcs import WCS

from bliss.datasets.galsim_galaxies import load_psf_from_file
from bliss.datasets.sdss import get_flux_coadd, get_hlr_coadd
from bliss.metrics import ClassificationMetrics, DetectionMetrics


def add_extra_coadd_info(coadd_cat_file: str, psf_image_file: str, pixel_scale: float, wcs: WCS):
    """Add additional useful information to coadd catalog."""
    coadd_cat = Table.read(coadd_cat_file)

    if not {"x", "y", "galaxy_bool", "flux", "mag", "hlr"}.issubset(set(coadd_cat.columns)):

        psf = load_psf_from_file(psf_image_file, pixel_scale)
        x, y = wcs.all_world2pix(coadd_cat["ra"], coadd_cat["dec"], 0)
        galaxy_bool = ~coadd_cat["probpsf"].data.astype(bool)
        is_saturated = coadd_cat["is_saturated"].data.astype(bool)
        flux, mag = get_flux_coadd(coadd_cat)
        hlr = get_hlr_coadd(coadd_cat, psf)

        # add info to catalog directly
        coadd_cat["x"] = x
        coadd_cat["y"] = y
        coadd_cat["galaxy_bool"] = galaxy_bool
        coadd_cat["is_saturated"] = is_saturated
        coadd_cat["flux"] = flux
        coadd_cat["mag"] = mag
        coadd_cat["hlr"] = hlr

        coadd_cat.write(coadd_cat_file, overwrite=True)  # overwrite with additional info.


def get_params_from_coadd(coadd_cat: str, bp: int, h: int, w: int):
    """Load coadd catalog from file, add extra useful information, convert to tensors."""
    assert {"x", "y", "galaxy_bool", "flux", "mag", "hlr"}.issubset(set(coadd_cat.columns))

    # filter saturated objects
    coadd_cat = coadd_cat[~coadd_cat["is_saturated"]]

    # filter objects in border or beyond.
    # NOTE: This assumes tiling scheme used in `predict.py`
    x, y = coadd_cat["x"], coadd_cat["y"]
    coadd_cat = coadd_cat[(x > bp) & (x < w - bp) & (y > bp) & (y < h - bp)]

    # extract all information we need for metrics.
    x = torch.from_numpy(coadd_cat["x"].data).reshape(-1, 1)
    y = torch.from_numpy(coadd_cat["y"].data).reshape(-1, 1)
    locs = torch.hstack((x, y)).reshape(-1, 2)

    galaxy_bool = torch.from_numpy(coadd_cat["galaxy_bool"].data).bool().reshape(-1)
    is_saturated = torch.from_numpy(coadd_cat["is_saturated"].data).bool().reshape(-1)

    flux = torch.from_numpy(coadd_cat["flux"].data).reshape(-1)
    mag = torch.from_numpy(coadd_cat["mag"].data).reshape(-1)

    return {
        "n_sources": len(locs),
        "plocs": locs,
        "galaxy_bool": galaxy_bool,
        "is_saturated": is_saturated,
        "flux": flux,
        "mag": mag,
    }


def apply_mag_cut(params: dict, mag_cut=25.0):
    """Apply magnitude cut to given parameters."""
    assert "mag" in params
    keep = params["mag"] < mag_cut
    return {k: v[keep] for k, v in params.items()}


def coadd_metrics(coadd_cat, est_params: dict, bp: int, h: int, w: int, mag_cut=25.0, slack=1.0):
    """Metrics based on usign coadd catalog as truth."""

    # extract 'true' parameters based on coadd catalog.
    true_params = get_params_from_coadd(coadd_cat, bp, h, w)

    # apply magnitude specified magnitude_cut
    true_params = apply_mag_cut(true_params, mag_cut)
    est_params = apply_mag_cut(true_params, mag_cut)

    detection_metrics = DetectionMetrics(slack)
    classification_metrics = ClassificationMetrics(slack)

    # collect params
    ntrue, nest = true_params["n_sources"], est_params["n_sources"]
    tlocs, elocs = true_params["plocs"], est_params["plocs"]
    tgbool, egbool = true_params["galaxy_bool"], est_params["galaxy_bool"]

    # update
    detection_metrics.update_single(ntrue, nest, tlocs, elocs)
    classification_metrics.update_single(ntrue, nest, tlocs, elocs, tgbool, egbool)

    # compute and return results
    return {**detection_metrics.compute(), **classification_metrics.compute()}
