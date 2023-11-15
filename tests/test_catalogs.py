from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate

from bliss.catalog import FullCatalog, SourceType, TileCatalog
from bliss.surveys.decals import DarkEnergyCameraLegacySurvey as DECaLS
from bliss.surveys.decals import TractorFullCatalog

# TODO: Add PhotoFullCatalog-specific tests (like loading, restricting by RA/DEC, downloading)


@pytest.fixture(scope="module")
def basic_tilecat():
    d = {
        "n_sources": torch.tensor([[[1, 1], [0, 1]]]),
        "locs": torch.zeros(1, 2, 2, 1, 2),
        "source_type": torch.ones((1, 2, 2, 1, 1)).bool(),
        "galaxy_params": torch.zeros((1, 2, 2, 1, 6)),
        "star_fluxes": torch.zeros((1, 2, 2, 1, 5)),
        "galaxy_fluxes": torch.zeros(1, 2, 2, 1, 5),
    }
    d["locs"][0, 0, 0, 0] = torch.tensor([0.5, 0.5])
    d["locs"][0, 0, 1, 0] = torch.tensor([0.5, 0.5])
    d["locs"][0, 1, 1, 0] = torch.tensor([0.5, 0.02])

    return TileCatalog(4, d)


@pytest.fixture(scope="module")
def multi_source_tilecat():
    d = {
        "n_sources": torch.tensor([[[2, 1], [0, 2]]]),
        "locs": torch.zeros(1, 2, 2, 2, 2),
        "source_type": torch.ones((1, 2, 2, 2, 1)).bool(),
        "galaxy_params": torch.zeros((1, 2, 2, 2, 6)),
        "star_fluxes": torch.zeros((1, 2, 2, 2, 5)),
        "galaxy_fluxes": torch.zeros(1, 2, 2, 2, 5),
    }
    d["galaxy_fluxes"][0, 0, 0, :, 2] = torch.tensor([1000, 500])
    d["galaxy_fluxes"][0, 0, 1, :, 2] = torch.tensor([10000, 200])
    d["galaxy_fluxes"][0, 1, 0, :, 2] = torch.tensor([0, 800])
    d["galaxy_fluxes"][0, 1, 1, :, 2] = torch.tensor([300, 600])

    return TileCatalog(4, d)


class TestBasicTileAndFullCatalogs:
    def test_unallowed_param(self):
        d_tile = {
            "n_sources": torch.zeros((1, 2, 2)),
            "locs": torch.zeros((1, 2, 2, 1, 2)),
            "unallowed": torch.zeros((1, 2, 2, 1, 2)),
        }
        with pytest.raises(ValueError):
            TileCatalog(4, d_tile)

        d_full = {
            "n_sources": torch.tensor([1]),
            "plocs": torch.tensor([5, 5]).reshape(1, 1, 2),
            "unallowed": torch.zeros(1, 1, 1),
        }
        with pytest.raises(ValueError):
            FullCatalog(10, 10, d_full)

    def test_param_accessors(self):
        d_tile = {
            "n_sources": torch.tensor([[[[1], [0]], [[1], [1]]]]).reshape((1, 2, 2)),
            "locs": torch.zeros((1, 2, 2, 1, 2)),
            "source_type": torch.tensor([[[1], [1]], [[1], [0]]]).reshape((1, 2, 2, 1, 1)),
        }
        tile_cat = TileCatalog(4, d_tile)
        assert tile_cat.locs.equal(tile_cat["locs"])
        assert tile_cat.galaxy_bools.equal(tile_cat["galaxy_bools"])

        keys = tile_cat.to_dict().keys()
        assert "locs" in keys
        assert "source_type" in keys
        assert "galaxy_bools" not in keys

        full_cat = tile_cat.to_full_params()
        assert full_cat.plocs.equal(full_cat["plocs"])
        assert full_cat.galaxy_bools.equal(full_cat["galaxy_bools"])

    def test_restrict_tile_cat_to_brightest(self, multi_source_tilecat):
        cat = multi_source_tilecat.get_brightest_sources_per_tile(band=2)
        assert cat.max_sources == 1
        assert cat.n_sources.max() == 1
        assert cat.n_sources.sum() == 3
        assert cat["galaxy_fluxes"].sum() == 11600.0
        assert cat["galaxy_fluxes"].max() == 10000.0

        # do it again to make sure nothing changes
        assert cat.get_brightest_sources_per_tile(band=2).max_sources == 1

    def test_filter_tile_cat_by_flux(self, multi_source_tilecat):
        cat = multi_source_tilecat.filter_tile_catalog_by_flux(300, 2000)
        assert cat.max_sources == 2
        assert cat.n_sources.sum() == 3
        assert cat["galaxy_fluxes"].sum() == 2100.0
        assert cat["galaxy_fluxes"].max() == 1000.0

    def test_bin_full_cat_by_flux(self):
        d = {
            "n_sources": torch.tensor([3]),
            "plocs": torch.tensor([[10, 10], [20, 20], [30, 30]]).reshape(1, 3, 2),
            "source_type": torch.tensor([1, 1, 0]).reshape(1, 3, 1),
            "mags": torch.tensor([20, 23, 22]).reshape(1, 3, 1),
        }
        binned_cat = FullCatalog(40, 40, d).apply_param_bin("mags", 21, 24)
        new_plocs = binned_cat.plocs[:, 0 : binned_cat.n_sources]
        new_mags = binned_cat["mags"][:, 0 : binned_cat.n_sources]

        assert new_plocs.shape == (1, 2, 2)
        assert new_mags.max() < 24
        assert new_mags.min() > 21

    def test_multiple_sources_one_tile(self):
        d = {
            "n_sources": torch.tensor([2]),
            "plocs": torch.tensor([[0.5, 0.5], [0.6, 0.6]]).reshape(1, 2, 2),
            "source_type": torch.full((1, 2, 1), SourceType.GALAXY),
        }
        full_cat = FullCatalog(2, 2, d)

        with pytest.raises(ValueError) as error_info:
            full_cat.to_tile_params(1, 1, ignore_extra_sources=False)
        assert error_info.value.args[0] == "# of sources per tile exceeds `max_sources_per_tile`."

        # should return only first source in first tile.
        tile_cat = full_cat.to_tile_params(1, 1, ignore_extra_sources=True)
        assert torch.equal(tile_cat.n_sources, torch.tensor([[[1, 0], [0, 0]]]))

        # test to_tile_coords and to_full_coords (set max_sources_per_tile to 2)
        convert_full_cat = full_cat.to_tile_params(1, 2).to_full_params()
        assert torch.allclose(convert_full_cat.plocs, full_cat.plocs)

        correct_locs = torch.tensor([[[0.5, 0.5], [0, 0]], [[0, 0], [0, 0]]]).reshape(1, 2, 2, 1, 2)
        assert torch.allclose(tile_cat.locs, correct_locs)

        correct_gbs = torch.tensor([[[1], [0]], [[0], [0]]]).reshape(1, 2, 2, 1, 1)
        assert torch.equal(tile_cat.galaxy_bools, correct_gbs)


class TestDecalsCatalog:
    def test_load_decals_from_file(self, cfg, monkeypatch):
        monkeypatch.setattr("bliss.surveys.decals.DECaLS_PSF.__init__", lambda *_args: None)

        brickname = "3366m010"
        sample_file = (
            Path(cfg.paths.decals) / brickname[:3] / brickname / f"tractor-{brickname}.fits"
        )
        the_cfg = cfg.copy()
        the_cfg.predict.dataset = cfg.surveys.decals
        the_cfg.encoder.bands = [DECaLS.BANDS.index("r")]
        decals = instantiate(the_cfg.predict.dataset)
        decals_cat = TractorFullCatalog.from_file(
            cat_path=sample_file,
            wcs=decals[0]["wcs"][DECaLS.BANDS.index("r")],
            height=decals[0]["background"].shape[1],
            width=decals[0]["background"].shape[2],
        )

        ras = decals_cat["ra"].numpy()
        decs = decals_cat["dec"].numpy()

        assert np.isclose(np.min(ras), 336.5, atol=1e-4)
        assert np.isclose(np.max(ras), 336.75, atol=1e-4)
        assert np.isclose(np.min(decs), -1.125, atol=1e-4)
        assert np.isclose(np.max(decs), -0.875, atol=1e-4)
