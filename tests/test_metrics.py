import numpy as np
import pytest
import torch
from hydra.utils import instantiate

from bliss.catalog import FullCatalog, TileCatalog
from bliss.metrics import BlissMetrics, MetricsMode, three_way_matching
from bliss.predict import align, prepare_image
from bliss.surveys.decals import DecalsFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS


class TestMetrics:
    def _get_sdss_data(self, cfg):
        """Loads SDSS frame and Photo Catalog."""
        sdss = instantiate(cfg.surveys.sdss)

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

        image = align(image, sdss[0]["wcs"], SDSS.BANDS.index("r"))
        background = align(background, sdss[0]["wcs"], SDSS.BANDS.index("r"))

        # crop to center fourth
        height, width = image[0].shape
        min_h, min_w = height // 4, width // 4
        max_h, max_w = min_h * 3, min_w * 3
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
        decals_cat = DecalsFullCatalog.from_file(decals_path, wcs, image.shape[1], image.shape[2])
        decals_cat = decals_cat.to(torch.device("cpu"))

        # get predicted BLISS catalog
        with torch.no_grad():
            batch = {
                "images": prepare_image(image, cfg.predict.device).float(),
                "background": prepare_image(background, cfg.predict.device).float(),
            }
            encoder.eval()
            encoder = encoder.float()
            pred = encoder.encode_batch(batch)
            bliss_cat = encoder.variational_mode(pred).to(torch.device("cpu"))

        bliss_cat.plocs += torch.tensor(
            [h_lim[0] + cfg.encoder.tile_slen, w_lim[0] + cfg.encoder.tile_slen]
        )  # coords in original image

        return {"decals": decals_cat, "photo": photo_cat, "bliss": bliss_cat}

    @pytest.fixture(scope="class")
    def tile_catalog(self, cfg, multiband_dataloader):
        """Generate a tile catalog for testing classification metrics."""
        tile_cat = next(iter(multiband_dataloader))["tile_catalog"]
        return TileCatalog(cfg.simulator.prior.tile_slen, tile_cat)

    def test_metrics(self):
        """Tests basic computations using simple toy data."""
        slen = 50
        metrics = BlissMetrics(mode=MetricsMode.FULL, slack=1.0)

        true_locs = torch.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.2, 0.2], [0.1, 0.1]]])
        est_locs = torch.tensor([[[0.49, 0.49], [0.1, 0.1]], [[0.19, 0.19], [0.01, 0.01]]])
        true_source_type = torch.tensor([[[1], [0]], [[1], [1]]])
        est_source_type = torch.tensor([[[0], [1]], [[1], [0]]])

        d_true = {
            "n_sources": torch.tensor([1, 2]),
            "plocs": true_locs * slen,
            "source_type": true_source_type,
        }
        true_params = FullCatalog(slen, slen, d_true)

        d_est = {
            "n_sources": torch.tensor([2, 2]),
            "plocs": est_locs * slen,
            "source_type": est_source_type,
        }
        est_params = FullCatalog(slen, slen, d_est)

        results = metrics(true_params, est_params)
        precision = results["detection_precision"]
        recall = results["detection_recall"]
        avg_distance = results["avg_distance"]

        class_acc = results["class_acc"]

        assert np.isclose(precision, 2 / (2 + 2))
        assert np.isclose(recall, 2 / 3)
        assert np.isclose(class_acc, 1 / 2)
        assert np.isclose(avg_distance, 50 * (0.01 + (0.01 + 0.09) / 2) / 2)

    def test_no_sources(self):
        """Tests that metrics work when there are no true or estimated sources."""
        metrics = BlissMetrics(mode=MetricsMode.FULL, slack=2.0)

        true_locs = torch.tensor(
            [[[10, 10]], [[20, 20]], [[30, 30]], [[40, 40]]], dtype=torch.float
        )
        est_locs = torch.tensor([[[10, 10]], [[20, 20]], [[30, 30]], [[41, 41]]], dtype=torch.float)
        true_source_type = torch.tensor([[[1]], [[1]], [[1]], [[1]]])
        est_source_type = torch.tensor([[[1]], [[1]], [[1]], [[1]]])
        true_sources = torch.tensor([0, 0, 1, 1])
        est_sources = torch.tensor([0, 1, 0, 1])

        d_true = {"n_sources": true_sources, "plocs": true_locs, "source_type": true_source_type}
        true_params = FullCatalog(50, 50, d_true)

        d_est = {"n_sources": est_sources, "plocs": est_locs, "source_type": est_source_type}
        est_params = FullCatalog(50, 50, d_est)

        results = metrics(true_params, est_params)

        assert results["detection_precision"] == 1 / 2
        assert results["detection_recall"] == 1 / 2
        assert results["gal_tp"] == 1
        assert results["avg_distance"] == 1

    def test_classification_metrics_tile(self, tile_catalog):
        """Test galaxy classification metrics on tile catalog."""
        metrics = BlissMetrics(mode=MetricsMode.TILE, slack=1.0)
        results = metrics(tile_catalog, tile_catalog)
        for metric in metrics.classification_metrics:
            if metric in {"gal_fluxes", "star_fluxes"}:
                for band in "ugriz":
                    assert results[f"{metric}_{band}_mae"] == 0
            else:
                assert results[f"{metric}_mae"] == 0

    def test_classification_metrics_full(self, tile_catalog):
        """Test galaxy classification metrics on full catalog."""
        full_catalog = tile_catalog.to_full_params()
        metrics = BlissMetrics(mode=MetricsMode.FULL, slack=1.0)
        results = metrics(full_catalog, full_catalog)
        for metric in metrics.classification_metrics:
            if metric in {"gal_fluxes", "star_fluxes"}:
                for band in "ugriz":
                    assert results[f"{metric}_{band}_mae"] == 0
            else:
                assert results[f"{metric}_mae"] == 0

    def test_photo_self_agreement(self, catalogs):
        """Compares PhotoFullCatalog to itself as safety check for metrics."""
        metrics = BlissMetrics(mode=MetricsMode.FULL, slack=1.0)
        results = metrics(catalogs["photo"], catalogs["photo"])
        assert results["f1"] == 1

    def test_decals_self_agreement(self, catalogs):
        """Compares Decals catalog to itself as safety check for metrics."""
        metrics = BlissMetrics(mode=MetricsMode.FULL, slack=1.0)
        results = metrics(catalogs["decals"], catalogs["decals"])
        assert results["f1"] == 1

    def test_photo_decals_agree(self, catalogs):
        """Compares metrics for agreement between Photo catalog and Decals catalog."""
        metrics = BlissMetrics(mode=MetricsMode.FULL, slack=1.0)
        results = metrics(catalogs["decals"], catalogs["photo"])
        assert results["detection_precision"] > 0.8

    def test_bliss_photo_agree(self, catalogs):
        """Compares metrics for agreement between BLISS-inferred catalog and Photo catalog."""
        slack = 1.0
        metrics = BlissMetrics(mode=MetricsMode.FULL, slack=slack)
        results = metrics(catalogs["photo"], catalogs["bliss"])
        assert results["f1"] > 0.7
        assert results["avg_keep_distance"] < slack

    def test_bliss_photo_agree_comp_decals(self, catalogs):
        """Compares metrics between BLISS and Photo catalog with DECaLS as GT."""
        decals_cat = catalogs["decals"]
        photo_cat = catalogs["photo"]
        bliss_cat = catalogs["bliss"]

        metrics = BlissMetrics(mode=MetricsMode.FULL, slack=1.0)

        bliss_vs_decals = metrics(decals_cat, bliss_cat)
        photo_vs_decals = metrics(decals_cat, photo_cat)

        assert bliss_vs_decals["f1"] > photo_vs_decals["f1"]

    def test_three_way_matching(self):
        gt = {
            "plocs": torch.arange(5, 30, 5, dtype=float).reshape(1, -1, 1).repeat(1, 1, 2) + 0.5,
            "n_sources": torch.tensor([5]),
        }
        pred = {"plocs": gt["plocs"].clone(), "n_sources": gt["n_sources"].clone()}
        pred["plocs"] = pred["plocs"][:, :-1]
        pred["plocs"][0, 2] = torch.tensor([20.5, 15.5])
        pred["n_sources"] = torch.tensor([4])

        comp = {"plocs": gt["plocs"].clone(), "n_sources": gt["n_sources"].clone()}
        comp["plocs"][0, 1] = torch.tensor([10.5, 20.5])
        comp["plocs"] = comp["plocs"][:, :-1]
        comp["n_sources"] = torch.tensor([4])

        gt_cat = FullCatalog(30, 30, gt)
        pred_cat = FullCatalog(30, 30, pred)
        comp_cat = FullCatalog(30, 30, comp)

        matches = three_way_matching(pred_cat, comp_cat, gt_cat)

        assert matches["gt_all"] == {0, 1, 2, 3}
        assert matches["gt_pred_only"] == {1}
        assert matches["gt_comp_only"] == {2}
        assert matches["pred_only"] == {2}
        assert matches["comp_only"] == {1}
