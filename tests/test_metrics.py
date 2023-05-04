from pathlib import Path

import torch
from astropy.io import fits
from astropy.wcs import WCS

from bliss.catalog import FullCatalog
from bliss.metrics import ClassificationMetrics, DetectionMetrics
from bliss.surveys.decals import DecalsFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog, SloanDigitalSkySurvey

RA_LIM = (336.62564677, 336.64428049)
DEC_LIM = (-0.96927915, -0.95064804)


def get_decals_data(filename, wcs=None):
    """Helper function to load DECaLS data for test cases."""

    cat = DecalsFullCatalog.from_file(filename, RA_LIM, DEC_LIM)

    # if provided, use WCS to convert RA and DEC to plocs
    if wcs is not None:
        plocs = cat.get_plocs_from_ra_dec(wcs)
        cat.plocs = plocs
        cat.height, cat.width = wcs.array_shape

    return cat


def constrain_photo_cat(photo_cat):
    """Helper function to restrict photo catalog to within RA and DEC limits."""
    ra = photo_cat["ra"].numpy().squeeze()
    dec = photo_cat["dec"].numpy().squeeze()

    keep = (ra > RA_LIM[0]) & (ra < RA_LIM[1]) & (dec >= DEC_LIM[0]) & (dec <= DEC_LIM[1])
    plocs = photo_cat.plocs[:, keep]
    galaxy_bools = photo_cat["galaxy_bools"][:, keep]
    n_sources = torch.tensor([plocs.size()[1]])

    d = {"plocs": plocs, "n_sources": n_sources, "galaxy_bools": galaxy_bools}
    return PhotoFullCatalog(photo_cat.height, photo_cat.width, d)


def test_metrics():
    """Tests basic metrics using simple toy data."""
    slen = 50
    slack = 1.0
    detect = DetectionMetrics(slack)
    classify = ClassificationMetrics(slack)

    true_locs = torch.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.2, 0.2], [0.1, 0.1]]])
    est_locs = torch.tensor([[[0.49, 0.49], [0.1, 0.1]], [[0.19, 0.19], [0.01, 0.01]]])
    true_galaxy_bools = torch.tensor([[[1], [0]], [[1], [1]]])
    est_galaxy_bools = torch.tensor([[[0], [1]], [[1], [0]]])

    d_true = {
        "n_sources": torch.tensor([1, 2]),
        "plocs": true_locs * slen,
        "galaxy_bools": true_galaxy_bools,
    }
    true_params = FullCatalog(slen, slen, d_true)

    d_est = {
        "n_sources": torch.tensor([2, 2]),
        "plocs": est_locs * slen,
        "galaxy_bools": est_galaxy_bools,
    }
    est_params = FullCatalog(slen, slen, d_est)

    results_detection = detect(true_params, est_params)
    precision = results_detection["precision"]
    recall = results_detection["recall"]
    avg_distance = results_detection["avg_distance"]

    results_classify = classify(true_params, est_params)
    class_acc = results_classify["class_acc"]
    conf_matrix = results_classify["conf_matrix"]

    assert precision == 2 / (2 + 2)
    assert recall == 2 / 3
    assert class_acc == 1 / 2
    assert conf_matrix.eq(torch.tensor([[1, 1], [0, 0]])).all()
    assert avg_distance.item() == 50 * (0.01 + (0.01 + 0.09) / 2) / 2


def test_photo_self_agreement(cfg):
    """Compares PhotoFullCatalog to itself as safety check for metrics."""
    slack = 1.0
    detect = DetectionMetrics(slack)

    sdss_path = cfg.paths.sdss
    photo_cat = PhotoFullCatalog.from_file(sdss_path, run=94, camcol=1, field=12, band=2)  # R band

    results_detection = detect(photo_cat, photo_cat)
    f1_score = results_detection["f1"]
    assert f1_score == 1


def test_decals_self_agreement(cfg):
    """Compares Decals catalog to itself as safety check for metrics."""
    slack = 1.0
    detect = DetectionMetrics(slack)

    # load file and WCS
    data_file = Path(cfg.paths.decals).joinpath("tractor-3366m010.fits")
    image_file = Path(cfg.paths.decals).joinpath("cutout_336.635_-0.9600.fits")

    with fits.open(image_file) as f:
        wcs = WCS(f[0].header)  # pylint: disable=E1101
        true_params = get_decals_data(data_file, wcs)

    results_detection = detect(true_params, true_params)
    f1_score = results_detection["f1"]
    assert f1_score == 1


def test_photo_decals_agree(cfg):
    """Compares metrics for agreement between Photo catalog and Decals catalog."""
    slack = 1.0
    detect = DetectionMetrics(slack)

    decals_file = Path(cfg.paths.decals).joinpath("tractor-3366m010.fits")
    sdss_path = cfg.paths.sdss

    # load SDSS WCS and use to get plocs for DECaLS catalog
    sdss = SloanDigitalSkySurvey(sdss_path, 94, 1, fields=(12,), bands=(2,))
    wcs: WCS = sdss[0]["wcs"][0]
    decals_cat = get_decals_data(decals_file, wcs)

    # get photo catalog constrained to region
    photo_cat = PhotoFullCatalog.from_file(sdss_path, run=94, camcol=1, field=12, band=2)  # R band
    photo_cat = constrain_photo_cat(photo_cat)

    results_detection = detect(decals_cat, photo_cat)
    precision = results_detection["precision"]
    assert precision == 1


def test_bliss_photo_agree():
    """Compares metrics for agreement between BLISS-inferred catalog and Photo catalog."""
    raise NotImplementedError()


def test_bliss_photo_agree_comp_decals():
    """Tests that metrics agree between BLISS and Photo catalog when computed with decals as GT."""
    raise NotImplementedError()


def test_bliss_hst_agree():
    """Compares metrics for agreement between BLISS-inferred catalog and Hst catalog."""
    raise NotImplementedError()
