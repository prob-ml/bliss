from pathlib import Path

import pytest
import torch

from bliss.catalog import FullCatalog, SourceType, TileCatalog

# TODO: Add PhotoFullCatalog-specific tests (like loading, restricting by RA/DEC, downloading)


@pytest.fixture(scope="module")
def basic_tilecat():
    d = {
        "n_sources": torch.tensor([[[1, 1], [0, 1]]]),
        "locs": torch.zeros(1, 2, 2, 1, 2),
        "source_type": torch.ones((1, 2, 2, 1, 1)).bool(),
        "galaxy_params": torch.zeros((1, 2, 2, 1, 6)),
        "fluxes": torch.zeros((1, 2, 2, 1, 5)),
    }
    d["locs"][0, 0, 0, 0] = torch.tensor([0.5, 0.5])
    d["locs"][0, 0, 1, 0] = torch.tensor([0.5, 0.5])
    d["locs"][0, 1, 1, 0] = torch.tensor([0.5, 0.02])

    return TileCatalog(d)


@pytest.fixture(scope="module")
def multi_source_tilecat():
    d = {
        "n_sources": torch.tensor([[[2, 1], [0, 2]]]),
        "locs": torch.zeros(1, 2, 2, 2, 2),
        "source_type": torch.ones((1, 2, 2, 2, 1)).bool(),
        "galaxy_params": torch.zeros((1, 2, 2, 2, 6)),
        "fluxes": torch.zeros(1, 2, 2, 2, 5),
    }
    d["fluxes"][0, 0, 0, :, 2] = torch.tensor([1000, 500])
    d["fluxes"][0, 0, 1, :, 2] = torch.tensor([10000, 200])
    d["fluxes"][0, 1, 0, :, 2] = torch.tensor([0, 800])
    d["fluxes"][0, 1, 1, :, 2] = torch.tensor([300, 600])

    return TileCatalog(d)


@pytest.fixture(scope="module")
def multi_source_fullcat():
    d = {
        "n_sources": torch.tensor([2, 3, 1]),
        "plocs": torch.zeros((3, 3, 2)),
        "source_type": torch.ones((3, 3, 1)).bool(),
        "fluxes": torch.zeros(3, 3, 6),
    }

    d["plocs"][0, 0, :] = torch.tensor([300, 600])
    d["plocs"][0, 1, :] = torch.tensor([1200, 1300])
    d["plocs"][1, 0, :] = torch.tensor([730, 73])
    d["plocs"][1, 1, :] = torch.tensor([1500, 1600])
    d["plocs"][1, 2, :] = torch.tensor([999, 998])
    d["plocs"][2, 0, :] = torch.tensor([1999, 1977])

    d["fluxes"][0, :, 2] = torch.tensor([1000, 500, 0])
    d["fluxes"][1, :, 2] = torch.tensor([10000, 545, 123])
    d["fluxes"][2, :, 2] = torch.tensor([124, 0, 0])

    return FullCatalog(2000, 2000, d)


class TestBasicTileAndFullCatalogs:
    def test_param_accessors(self):
        d_tile = {
            "n_sources": torch.tensor([[[[1], [0]], [[1], [1]]]]).reshape((1, 2, 2)),
            "locs": torch.zeros((1, 2, 2, 1, 2)),
            "source_type": torch.tensor([[[1], [1]], [[1], [0]]]).reshape((1, 2, 2, 1, 1)),
        }
        tile_cat = TileCatalog(d_tile)

        keys = tile_cat.keys()
        assert "locs" in keys
        assert "source_type" in keys
        assert "galaxy_bools" not in keys

        full_cat = tile_cat.to_full_catalog(4)
        keys = full_cat.keys()
        assert "plocs" in keys

    def test_restrict_tile_cat_to_brightest(self, multi_source_tilecat):
        cat = multi_source_tilecat.get_brightest_sources_per_tile(band=2)
        assert cat.max_sources == 1
        assert cat["n_sources"].max() == 1
        assert cat["n_sources"].sum() == 3
        assert cat["fluxes"].sum() == 11600.0
        assert cat["fluxes"].max() == 10000.0

        # do it again to make sure nothing changes
        assert cat.get_brightest_sources_per_tile(band=2).max_sources == 1

    def test_filter_tile_cat_by_flux(self, multi_source_tilecat):
        cat = multi_source_tilecat.filter_by_flux(300)
        assert cat.max_sources == 2
        assert cat["n_sources"].sum() == 4
        r_band_flux = cat["fluxes"][..., 2:3]
        r_band_flux = torch.where(cat.galaxy_bools, r_band_flux, torch.inf)
        assert r_band_flux.min().item() == 500

    def test_bin_full_cat_by_flux(self):
        d = {
            "n_sources": torch.tensor([3]),
            "plocs": torch.tensor([[10, 10], [20, 20], [30, 30]]).reshape(1, 3, 2),
            "source_type": torch.tensor([1, 1, 0]).reshape(1, 3, 1),
            "mags": torch.tensor([20, 23, 22]).reshape(1, 3, 1),
        }
        binned_cat = FullCatalog(40, 40, d).apply_param_bin("mags", 21, 24)
        new_plocs = binned_cat["plocs"][:, 0 : binned_cat["n_sources"]]
        new_mags = binned_cat["mags"][:, 0 : binned_cat["n_sources"]]

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
            full_cat.to_tile_catalog(1, 1, ignore_extra_sources=False)
        assert error_info.value.args[0] == "# of sources per tile exceeds `max_sources_per_tile`."

        # should return only first source in first tile.
        tile_cat = full_cat.to_tile_catalog(1, 1, ignore_extra_sources=True)
        assert torch.equal(tile_cat["n_sources"], torch.tensor([[[1, 0], [0, 0]]]))

        # test to_tile_coords and to_full_coords
        tile_slen = 1
        max_sources = 2
        fc_converted = full_cat.to_tile_catalog(tile_slen, max_sources).to_full_catalog(tile_slen)
        assert torch.allclose(fc_converted["plocs"], full_cat["plocs"])

        correct_locs = torch.tensor([[[0.5, 0.5], [0, 0]], [[0, 0], [0, 0]]]).reshape(1, 2, 2, 1, 2)
        assert torch.allclose(tile_cat["locs"], correct_locs)

        correct_gbs = torch.tensor([[[1], [0]], [[0], [0]]]).reshape(1, 2, 2, 1, 1)
        assert torch.equal(tile_cat.galaxy_bools, correct_gbs)

    def test_filter_full_catalog_by_ploc_box(self, multi_source_fullcat):
        cat = multi_source_fullcat.filter_by_ploc_box(torch.tensor([0.0, 0.0]), 1000.0)
        assert torch.equal(cat["n_sources"], torch.tensor([1, 2, 0]))
        assert cat["plocs"].shape[1] == 2
        assert torch.allclose(cat["plocs"][0, 0, :], torch.tensor([300.0, 600.0]))
        assert torch.allclose(cat["plocs"][1, 0, :], torch.tensor([730.0, 73.0]))
        assert torch.allclose(cat["plocs"][1, 1, :], torch.tensor([999.0, 998.0]))
        assert cat["fluxes"].shape[1] == 2
        assert torch.allclose(cat["fluxes"][0, :, 2], torch.tensor([1000.0, 500.0]))
        assert torch.allclose(cat["fluxes"][1, :, 2], torch.tensor([10000.0, 123.0]))
        assert torch.allclose(cat["fluxes"][2, :, 2], torch.tensor([124.0, 0.0]))

    def test_tile_full_round_trip(self, cfg):
        with open(Path(cfg.paths.test_data) / "sdss_preds.pt", "rb") as f:
            test_cat = torch.load(f)

        # we'll do a "round trip" test: convert the catalog to a full catalog and back
        true_tile_cat0 = TileCatalog(test_cat)
        true_full_cat = true_tile_cat0.to_full_catalog(cfg.decoder.tile_slen)
        true_tile_cat = true_full_cat.to_tile_catalog(
            tile_slen=cfg.decoder.tile_slen,
            max_sources_per_tile=cfg.prior.max_sources,
            ignore_extra_sources=True,
        )

        # fields only need to match if a source is present
        assert (true_tile_cat0["n_sources"] == true_tile_cat["n_sources"]).all()
        assert true_tile_cat.max_sources == 1
        gating = true_tile_cat0["n_sources"].unsqueeze(-1).unsqueeze(-1)
        keys_to_match = true_tile_cat0.keys() - "n_sources"
        for k in keys_to_match:
            v0 = true_tile_cat0[k] * gating
            v1 = true_tile_cat[k] * gating
            assert torch.isclose(v0, v1, rtol=1e-4, atol=1e-6).all()
