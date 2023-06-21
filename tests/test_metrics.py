import numpy as np
import pytest
import torch
from hydra.utils import instantiate

from bliss.catalog import FullCatalog, TileCatalog
from bliss.metrics import BlissMetrics
from bliss.predict import align, prepare_image
from bliss.surveys.decals import DecalsFullCatalog
from bliss.surveys.sdss import PhotoFullCatalog, SloanDigitalSkySurvey


class TestMetrics:
    def _get_sdss_data(self, cfg):
        """Loads SDSS frame and Photo Catalog."""
        photo_cat = PhotoFullCatalog.from_file(
            cfg.paths.sdss,
            run=cfg.predict.dataset.run,
            camcol=cfg.predict.dataset.camcol,
            field=cfg.predict.dataset.fields[0],
            band=cfg.predict.dataset.bands[0],
        )
        sdss = SloanDigitalSkySurvey(cfg.paths.sdss, 94, 1, (12,), (0, 1, 2, 3, 4))

        return photo_cat, sdss

    def _get_image_and_background(self, sdss):
        """Aligns, crops image and background from SDSS frame to reduce size."""
        image = sdss[0]["image"]
        background = sdss[0]["background"]

        image = align(image, sdss)
        background = align(background, sdss)

        # crop to center fourth
        height, width = image[0].shape
        min_h, min_w = height // 4, width // 4
        max_h, max_w = min_h * 3, min_w * 3
        cropped_image = image[:, min_h:max_h, min_w:max_w]
        cropped_background = background[:, min_h:max_h, min_w:max_w]

        return cropped_image, cropped_background, (min_w, max_w), (min_h, max_h)

    def _get_photo_cat(self, photo_cat, ra_lim, dec_lim):
        """Helper function to restrict photo catalog to within RA and DEC limits."""
        ra = photo_cat["ra"].squeeze()
        dec = photo_cat["dec"].squeeze()

        keep = (ra > ra_lim[0]) & (ra < ra_lim[1]) & (dec >= dec_lim[0]) & (dec <= dec_lim[1])
        plocs = photo_cat.plocs[:, keep]
        n_sources = torch.tensor([plocs.size()[1]])

        d = {"plocs": plocs, "n_sources": n_sources}
        for key in photo_cat.keys():
            d[key] = photo_cat[key][:, keep]

        return PhotoFullCatalog(
            plocs[0, :, 0].max() - plocs[0, :, 0].min(),  # new height
            plocs[0, :, 1].max() - plocs[0, :, 1].min(),  # new width
            d,
        )

    def _get_decals_cat(self, filename, ra_lim, dec_lim, wcs):
        """Helper function to load DECaLS data for test cases."""
        cat = DecalsFullCatalog.from_file(filename, ra_lim, dec_lim)

        # if provided, use WCS to convert RA and DEC to plocs
        if wcs is not None:
            plocs = cat.get_plocs_from_ra_dec(wcs)
            cat.plocs = plocs
            cat.height, cat.width = wcs.array_shape

        return cat

    @pytest.fixture(scope="class")
    def catalogs(self, cfg, encoder):
        """The main entry point to get data for most of the tests."""
        # load SDSS catalog and WCS
        cfg.predict.dataset.bands = [2]
        base_photo_cat, sdss = self._get_sdss_data(cfg)

        wcs = sdss[0]["wcs"][2]
        image, background, w_lim, h_lim = self._get_image_and_background(sdss)

        # get RA/DEC limits of cropped image and construct catalogs
        ra_lim, dec_lim = wcs.all_pix2world(w_lim, h_lim, 0)
        photo_cat = self._get_photo_cat(base_photo_cat, ra_lim, dec_lim).to(torch.device("cpu"))
        decals_path = cfg.predict.decals_frame
        decals_cat = self._get_decals_cat(decals_path, ra_lim, dec_lim, wcs).to(torch.device("cpu"))

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

    def _get_sliced_catalog(self, catalog, idx_to_keep):
        """Creates a new FullCatalog using only certain indices from old catalog."""
        d = {key: val[:, idx_to_keep, :] for key, val in catalog.items()}
        d["n_sources"] = torch.tensor([len(idx_to_keep)])
        d["plocs"] = catalog.plocs[:, idx_to_keep, :]
        return FullCatalog(catalog.height, catalog.width, d)

    @pytest.fixture(scope="class")
    def brightest_catalogs(self, catalogs):
        """Get catalogs restricted to only the brightest n sources."""
        decals_cat = catalogs["decals"]
        photo_cat = catalogs["photo"]
        bliss_cat = catalogs["bliss"]

        n = min(decals_cat.n_sources.item(), photo_cat.n_sources.item(), bliss_cat.n_sources.item())

        top_n_decals = torch.argsort(decals_cat["fluxes"].squeeze())[-n:]
        top_n_photo = torch.argsort(photo_cat["fluxes"].squeeze())[-n:]
        bliss_fluxes = (
            bliss_cat["star_fluxes"] * (bliss_cat["source_type"] is False)  # noqa: E712
            + bliss_cat["galaxy_params"][:, :, 0, None] * bliss_cat["source_type"]
        )  # galaxy fluxes
        top_n_bliss = torch.argsort(torch.sum(bliss_fluxes, dim=2).squeeze())[-n:]

        decals_cat = self._get_sliced_catalog(decals_cat, top_n_decals)
        photo_cat = self._get_sliced_catalog(photo_cat, top_n_photo)
        bliss_cat = self._get_sliced_catalog(bliss_cat, top_n_bliss)

        return {"decals": decals_cat, "photo": photo_cat, "bliss": bliss_cat}

    def test_metrics(self):
        """Tests basic computations using simple toy data."""
        slen = 50
        metrics = BlissMetrics(mode=BlissMetrics.Mode.Full, slack=1.0)

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
        metrics = BlissMetrics(mode=BlissMetrics.Mode.Full, slack=2.0)

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

    def test_classification_metrics(self, cfg):
        """Test galaxy classification metrics."""
        simulator = instantiate(cfg.simulator, prior={"batch_size": 4})
        batch = next(iter(simulator.train_dataloader()))
        catalog = TileCatalog(cfg.encoder.tile_slen, batch["tile_catalog"])
        metrics = BlissMetrics(mode=BlissMetrics.Mode.Tile, slack=1.0)
        results = metrics(catalog, catalog)
        for metric in metrics.classification_metrics:
            assert results[f"{metric}_mae"] == 0

    def test_photo_self_agreement(self, catalogs):
        """Compares PhotoFullCatalog to itself as safety check for metrics."""
        metrics = BlissMetrics(mode=BlissMetrics.Mode.Full, slack=1.0)
        results = metrics(catalogs["photo"], catalogs["photo"])
        assert results["f1"] == 1

    def test_decals_self_agreement(self, catalogs):
        """Compares Decals catalog to itself as safety check for metrics."""
        metrics = BlissMetrics(mode=BlissMetrics.Mode.Full, slack=1.0)
        results = metrics(catalogs["decals"], catalogs["decals"])
        assert results["f1"] == 1

    def test_photo_decals_agree(self, catalogs):
        """Compares metrics for agreement between Photo catalog and Decals catalog."""
        metrics = BlissMetrics(mode=BlissMetrics.Mode.Full, slack=1.0)
        results = metrics(catalogs["decals"], catalogs["photo"])
        assert results["detection_precision"] > 0.8

    def test_bliss_photo_agree(self, brightest_catalogs):
        """Compares metrics for agreement between BLISS-inferred catalog and Photo catalog."""
        slack = 1.0
        metrics = BlissMetrics(mode=BlissMetrics.Mode.Full, slack=slack)
        results = metrics(brightest_catalogs["photo"], brightest_catalogs["bliss"])
        assert results["f1"] > 0.7
        assert results["avg_keep_distance"] < slack

    def test_bliss_photo_agree_comp_decals(self, brightest_catalogs):
        """Compares metrics between BLISS and Photo catalog with DECaLS as GT."""
        decals_cat = brightest_catalogs["decals"]
        photo_cat = brightest_catalogs["photo"]
        bliss_cat = brightest_catalogs["bliss"]

        metrics = BlissMetrics(mode=BlissMetrics.Mode.Full, slack=1.0)

        bliss_vs_decals = metrics(decals_cat, bliss_cat)
        photo_vs_decals = metrics(decals_cat, photo_cat)

        assert np.isclose(bliss_vs_decals["f1"], photo_vs_decals["f1"], atol=0.1)
