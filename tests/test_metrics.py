import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import open_dict

from bliss.catalog import FullCatalog, TileCatalog
from bliss.encoder.metrics import (
    CatalogMatcher,
    DetectionPerformance,
    FluxError,
    SourceTypeAccuracy,
)
from bliss.surveys.decals import TractorFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog


class TestMetrics:
    def _get_sdss_data(self, cfg):
        """Loads SDSS frame and Photo Catalog."""
        cfg = cfg.copy()
        with open_dict(cfg):
            cfg.surveys.sdss.align_to_band = 2
        sdss = instantiate(cfg.surveys.sdss, load_image_data=True)
        sdss.prepare_data()

        run, camcol, field = sdss.image_id(0)
        photo_cat = PhotoFullCatalog.from_file(
            cat_path=cfg.paths.sdss
            + f"/{run}/{camcol}/{field}/photoObj-{run:06d}-{camcol}-{field:04d}.fits",
            wcs=sdss[0]["wcs"][cfg.simulator.prior.reference_band],
            height=sdss[0]["image"].shape[1],
            width=sdss[0]["image"].shape[2],
        )
        return photo_cat, sdss

    def _get_image_and_background(self, sdss):
        """Aligns, crops image and background from SDSS frame to reduce size."""
        image = sdss[0]["image"]
        background = sdss[0]["background"]

        # crop to center fourth
        height, width = image[0].shape
        min_h, min_w = height // 4, width // 4
        max_h, max_w = min_h * 3 - 8, min_w * 3
        cropped_image = image[:, min_h:max_h, min_w:max_w]
        cropped_background = background[:, min_h:max_h, min_w:max_w]

        return cropped_image, cropped_background, (min_w, max_w), (min_h, max_h)

    @pytest.fixture(scope="class")
    def catalogs(self, cfg, encoder):
        """The main entry point to get data for most of the tests."""
        # load SDSS catalog and WCS
        base_photo_cat, sdss = self._get_sdss_data(cfg)
        wcs = sdss[0]["wcs"][2]
        image, background, w_lim, h_lim = self._get_image_and_background(sdss)

        # get RA/DEC limits of cropped image and construct d
        ra_lim, dec_lim = wcs.all_pix2world(w_lim, h_lim, 0)
        photo_cat = base_photo_cat.restrict_by_ra_dec(ra_lim, dec_lim).to(torch.device("cpu"))
        decals_path = cfg.predict.decals_frame
        decals_cat = TractorFullCatalog.from_file(decals_path, wcs, image.shape[1], image.shape[2])
        decals_cat = decals_cat.to(torch.device("cpu"))

        # get predicted BLISS catalog
        def prep_image(x):  # noqa: WPS430
            return torch.from_numpy(x).float().unsqueeze(0).to(device=cfg.predict.device)

        with torch.no_grad():
            batch = {
                "images": prep_image(image),
                "background": prep_image(background),
                "psf_params": sdss[0]["psf_params"],
            }
            encoder.eval()
            encoder = encoder.float()
            bliss_cat = encoder.sample(batch, use_mode=True)
            bliss_cat = bliss_cat.to(torch.device("cpu")).to_full_catalog()

        bliss_cat["plocs"] += torch.tensor([h_lim[0], w_lim[0]])

        return {"decals": decals_cat, "photo": photo_cat, "bliss": bliss_cat}

    @pytest.fixture(scope="class")
    def tile_catalog(self, cfg, multiband_dataloader):
        """Generate a tile catalog for testing classification metrics."""
        tile_cat = next(iter(multiband_dataloader))["tile_catalog"]
        return TileCatalog(cfg.simulator.prior.tile_slen, tile_cat)

    def test_metrics(self):
        """Tests basic computations using simple toy data."""
        slen = 50

        true_locs = torch.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.2, 0.2], [0.1, 0.1]]])
        est_locs = torch.tensor([[[0.49, 0.49], [0.1, 0.1]], [[0.19, 0.19], [0.01, 0.01]]])
        true_source_type = torch.tensor([[[1], [0]], [[1], [1]]])
        est_source_type = torch.tensor([[[0], [1]], [[1], [0]]])

        d_true = {
            "n_sources": torch.tensor([1, 2]),
            "plocs": true_locs * slen,
            "source_type": true_source_type,
            "star_fluxes": torch.ones(2, 2, 5),
            "galaxy_fluxes": torch.ones(2, 2, 5),
            "galaxy_params": torch.ones(2, 2, 6),
        }
        true_params = FullCatalog(slen, slen, d_true)

        d_est = {
            "n_sources": torch.tensor([2, 2]),
            "plocs": est_locs * slen,
            "source_type": est_source_type,
            "star_fluxes": torch.ones(2, 2, 5),
            "galaxy_fluxes": torch.ones(2, 2, 5),
            "galaxy_params": torch.ones(2, 2, 6),
        }
        est_params = FullCatalog(slen, slen, d_est)

        matcher = CatalogMatcher(dist_slack=1.0, mag_band=2)
        matching = matcher.match_catalogs(true_params, est_params)

        detection_metrics = DetectionPerformance()
        dresults = detection_metrics(true_params, est_params, matching)
        assert np.isclose(dresults["detection_precision"], 2 / (2 + 2))
        assert np.isclose(dresults["detection_recall"], 2 / 3)

        acc_metrics = SourceTypeAccuracy(flux_bin_cutoffs=[200, 400, 600, 800, 1000])
        acc_results = acc_metrics(true_params, est_params, matching)
        assert np.isclose(acc_results["classification_acc"], 1 / 2)

    def test_no_sources(self):
        """Tests that metrics work when there are no true or estimated sources."""
        true_locs = torch.tensor(
            [[[10, 10]], [[20, 20]], [[30, 30]], [[40, 40]]], dtype=torch.float
        )
        est_locs = torch.tensor([[[10, 10]], [[20, 20]], [[30, 30]], [[41, 41]]], dtype=torch.float)
        true_source_type = torch.tensor([[[1]], [[1]], [[1]], [[1]]])
        est_source_type = torch.tensor([[[1]], [[1]], [[1]], [[1]]])
        true_sources = torch.tensor([0, 0, 1, 1])
        est_sources = torch.tensor([0, 1, 0, 1])

        d_true = {
            "n_sources": true_sources,
            "plocs": true_locs,
            "source_type": true_source_type,
            "star_fluxes": torch.ones(4, 1, 5),
            "galaxy_fluxes": torch.ones(4, 1, 5),
            "galaxy_params": torch.ones(4, 1, 6),
        }
        true_params = FullCatalog(50, 50, d_true)

        d_est = {
            "n_sources": est_sources,
            "plocs": est_locs,
            "source_type": est_source_type,
            "star_fluxes": torch.ones(4, 1, 5),
            "galaxy_fluxes": torch.ones(4, 1, 5),
            "galaxy_params": torch.ones(4, 1, 6),
        }
        est_params = FullCatalog(50, 50, d_est)

        matcher = CatalogMatcher(dist_slack=2.0, mag_band=2)
        matching = matcher.match_catalogs(true_params, est_params)

        detection_metrics = DetectionPerformance()
        dresults = detection_metrics(true_params, est_params, matching)

        assert dresults["detection_precision"] == 1 / 2
        assert dresults["detection_recall"] == 1 / 2

    def test_self_agreement(self, tile_catalog):
        """Test galaxy classification metrics on full catalog."""
        full_catalog = tile_catalog.to_full_catalog()

        matcher = CatalogMatcher(dist_slack=1.0, mag_band=2)
        matching = matcher.match_catalogs(full_catalog, full_catalog)

        detection_metrics = DetectionPerformance()
        dresults = detection_metrics(full_catalog, full_catalog, matching)
        assert dresults["detection_f1"] == 1

        acc_metrics = SourceTypeAccuracy(flux_bin_cutoffs=[200, 400, 600, 800, 1000])
        acc_results = acc_metrics(full_catalog, full_catalog, matching)
        assert acc_results["classification_acc"] == 1

        flux_metrics = FluxError("ugriz")
        flux_results = flux_metrics(full_catalog, full_catalog, matching)
        assert flux_results["flux_err_r_mae"] == 0

    def test_catalog_agreement(self, catalogs):
        """Compares catalogs as safety check for metrics."""
        matcher = CatalogMatcher(dist_slack=1.0)
        detection_metrics = DetectionPerformance(mag_band=None)

        pp_matching = matcher.match_catalogs(catalogs["photo"], catalogs["photo"])
        pp_results = detection_metrics(catalogs["photo"], catalogs["photo"], pp_matching)
        assert pp_results["detection_f1"] == 1

        dd_matching = matcher.match_catalogs(catalogs["decals"], catalogs["decals"])
        dd_results = detection_metrics(catalogs["decals"], catalogs["decals"], dd_matching)
        assert dd_results["detection_f1"] == 1

        dp_matching = matcher.match_catalogs(catalogs["decals"], catalogs["photo"])
        dp_results = detection_metrics(catalogs["decals"], catalogs["photo"], dp_matching)
        assert dp_results["detection_precision"] > 0.8

        # bliss finds many more sources than photo. recall here measures the fraction of sources
        # photo finds that bliss also finds
        pb_matching = matcher.match_catalogs(catalogs["photo"], catalogs["bliss"])
        pb_results = detection_metrics(catalogs["photo"], catalogs["bliss"], pb_matching)
        assert pb_results["detection_recall"] > 0.8

        # with the arguments reversed, precision below measures that same thing as recall did above
        bp_matching = matcher.match_catalogs(catalogs["bliss"], catalogs["photo"])
        bp_results = detection_metrics(catalogs["bliss"], catalogs["photo"], bp_matching)
        assert bp_results["detection_precision"] > 0.8

        db_matching = matcher.match_catalogs(catalogs["decals"], catalogs["bliss"])
        db_results = detection_metrics(catalogs["decals"], catalogs["bliss"], db_matching)
        assert db_results["detection_f1"] > dp_results["detection_f1"]
