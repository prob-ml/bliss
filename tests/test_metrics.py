import numpy as np
import pytest
import torch
from hydra.utils import instantiate

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
    @pytest.fixture(scope="class")
    def tile_catalog(self, cfg, multiband_dataloader):
        """Generate a tile catalog for testing classification metrics."""
        tile_cat = next(iter(multiband_dataloader))["tile_catalog"]
        return TileCatalog(tile_cat)

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
        full_catalog = tile_catalog.to_full_catalog(4)

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

    def test_photo_decals_catalogs_matches(self, cfg):
        """Compares catalogs as safety check for metrics."""
        sdss = instantiate(cfg.surveys.sdss, load_image_data=False)
        sdss.prepare_data()

        run, camcol, field = sdss.image_id(0)
        base_photo_cat = PhotoFullCatalog.from_file(
            cat_path=cfg.paths.sdss
            + f"/{run}/{camcol}/{field}/photoObj-{run:06d}-{camcol}-{field:04d}.fits",
            wcs=sdss[0]["wcs"][2],
            height=1488,
            width=2048,
        )

        wcs = sdss[0]["wcs"][2]

        w_lim, h_lim = ((512, 1536), (372, 1108))

        # get RA/DEC limits of cropped image and construct d
        ra_lim, dec_lim = wcs.all_pix2world(w_lim, h_lim, 0)
        photo_cat = base_photo_cat.restrict_by_ra_dec(ra_lim, dec_lim).to(torch.device("cpu"))
        decals_path = cfg.predict.decals_frame
        decals_cat = TractorFullCatalog.from_file(
            decals_path, wcs, h_lim[1] - h_lim[0], w_lim[1] - w_lim[0]
        )
        decals_cat = decals_cat.to(torch.device("cpu"))

        matcher = CatalogMatcher(dist_slack=1.0)
        detection_metrics = DetectionPerformance(mag_band=None)

        pp_matching = matcher.match_catalogs(photo_cat, photo_cat)
        pp_results = detection_metrics(photo_cat, photo_cat, pp_matching)
        assert pp_results["detection_f1"] == 1

        dd_matching = matcher.match_catalogs(decals_cat, decals_cat)
        dd_results = detection_metrics(decals_cat, decals_cat, dd_matching)
        assert dd_results["detection_f1"] == 1

        dp_matching = matcher.match_catalogs(decals_cat, photo_cat)
        dp_results = detection_metrics(decals_cat, photo_cat, dp_matching)
        assert dp_results["detection_precision"] > 0.8
