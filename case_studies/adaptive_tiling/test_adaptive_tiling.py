import warnings

import pytest
import torch
from hydra.utils import instantiate

from bliss.catalog import TileCatalog
from case_studies.adaptive_tiling.region_catalog import RegionCatalog, tile_cat_to_region_cat


class TestRegionEncoder:
    def test_region_loss(self, cfg):
        encoder = instantiate(cfg.region_encoder).to(cfg.predict.device)
        d = {
            "n_sources": torch.zeros(1, 7, 7),
            "locs": torch.zeros(1, 7, 7, 1, 2),
            "source_type": torch.zeros((1, 7, 7, 1, 1)).bool(),
            "galaxy_params": torch.zeros((1, 7, 7, 1, 6)),
            "star_fluxes": torch.zeros((1, 7, 7, 1, 5)),
            "galaxy_fluxes": torch.zeros(1, 7, 7, 1, 5),
        }
        region_cat = RegionCatalog(interior_slen=3.6, overlap_slen=0.4, d=d).to(cfg.predict.device)

        images = torch.rand(1, 5, 16, 16, device=cfg.predict.device)
        background = torch.rand(1, 5, 16, 16, device=cfg.predict.device)
        batch = {"images": images, "background": background}

        encoder.tiles_to_crop = 0
        encoder.eval()
        with torch.no_grad():
            x_cat_marginal, _ = encoder.get_marginal(batch)
            pred = encoder.get_predicted_dist(x_cat_marginal)
            encoder._get_loss(pred, region_cat)  # pylint: disable=W0212  # noqa: WPS437

    def test_region_locs_in_tiles(self):
        d = {
            "n_sources": torch.ones(1, 5, 5),
            "locs": torch.ones(1, 5, 5, 1, 2) * 0.6,
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cat = RegionCatalog(interior_slen=3.6, overlap_slen=0.4, d=d)

        interior_locs = cat.get_interior_locs_in_tile()
        locs_left, locs_right = cat.get_vertical_boundary_locs_in_tiles()
        locs_up, locs_down = cat.get_horizontal_boundary_locs_in_tiles()
        locs_ul, locs_ur, locs_bl, locs_br = cat.get_corner_locs_in_tiles()

        # interior
        assert torch.allclose(
            interior_locs[0, 0, 0], torch.tensor([0.6 * 3.8 / 4.2, 0.6 * 3.8 / 4.2])
        )
        assert torch.allclose(
            interior_locs[0, 0, 4], torch.tensor([0.6 * 3.8 / 4.2, (0.6 * 3.8 + 0.4) / 4.2])
        )
        assert torch.allclose(
            interior_locs[0, 2, 2],
            torch.tensor([(0.6 * 3.6 + 0.4) / 4.4, ((0.6 * 3.6 + 0.4) / 4.4)]),
        )

        # vertical boundary
        assert torch.allclose(
            locs_left[0, 0, 1], torch.tensor([0.6 * 3.8 / 4.2, (0.6 * 0.4 + 3.8) / 4.2])
        )
        assert torch.allclose(
            locs_right[0, 0, 1], torch.tensor([0.6 * 3.8 / 4.2, (0.6 * 0.4) / 4.4])
        )
        assert torch.allclose(
            locs_left[0, 4, 3], torch.tensor([(0.6 * 3.8 + 0.4) / 4.2, (0.6 * 0.4 + 4) / 4.4])
        )
        assert torch.allclose(
            locs_right[0, 4, 3], torch.tensor([(0.6 * 3.8 + 0.4) / 4.2, (0.6 * 0.4) / 4.2])
        )

        # horizontal boundary
        assert torch.allclose(
            locs_up[0, 1, 0], torch.tensor([(0.6 * 0.4 + 3.8) / 4.2, 0.6 * 3.8 / 4.2])
        )
        assert torch.allclose(locs_down[0, 1, 0], torch.tensor([0.6 * 0.4 / 4.4, 0.6 * 3.8 / 4.2]))
        assert torch.allclose(
            locs_up[0, 3, 2], torch.tensor([(0.6 * 0.4 + 4) / 4.4, (0.6 * 3.6 + 0.4) / 4.4])
        )
        assert torch.allclose(
            locs_down[0, 3, 2], torch.tensor([0.6 * 0.4 / 4.2, (0.6 * 3.6 + 0.4) / 4.4])
        )

        # corner
        assert torch.allclose(
            locs_ul[0, 1, 3], torch.tensor([(0.4 * 0.6 + 3.8) / 4.2, (0.4 * 0.6 + 4) / 4.4])
        )
        assert torch.allclose(
            locs_ur[0, 1, 3], torch.tensor([(0.4 * 0.6 + 3.8) / 4.2, 0.4 * 0.6 / 4.2])
        )
        assert torch.allclose(
            locs_bl[0, 1, 3], torch.tensor([0.4 * 0.6 / 4.4, (0.4 * 0.6 + 4) / 4.4])
        )
        assert torch.allclose(locs_br[0, 1, 3], torch.tensor([0.4 * 0.6 / 4.4, 0.4 * 0.6 / 4.2]))


class TestRegionCatalog:
    def test_properties(self, region_cat):
        assert region_cat.height == region_cat.width == 12
        assert region_cat.is_on_mask.sum() == 6
        assert torch.all(
            region_cat.interior_mask + region_cat.boundary_mask + region_cat.corner_mask,
        )
        assert not torch.any(
            region_cat.interior_mask * region_cat.boundary_mask * region_cat.corner_mask,
        )

    def test_region_coords(self, region_cat):
        coords = region_cat.get_region_coords()
        assert coords.amax(dim=(0, 1))[0] < region_cat.height
        assert coords.amax(dim=(0, 1))[1] < region_cat.width

    def test_region_sizes(self, region_cat):
        sizes = region_cat.get_region_sizes()
        assert sizes[0].equal(
            torch.tensor([[3.75, 3.75], [3.75, 0.5], [3.75, 3.5], [3.75, 0.5], [3.75, 3.75]])
        )
        assert sizes[1].equal(
            torch.tensor([[0.5, 3.75], [0.5, 0.5], [0.5, 3.5], [0.5, 0.5], [0.5, 3.75]])
        )
        assert torch.all(sizes[..., 0].sum(dim=0) == region_cat.height)
        assert torch.all(sizes[..., 1].sum(dim=1) == region_cat.width)

    def test_convert_to_full(self, region_cat):
        full_cat = region_cat.to_full_params()
        true_locs = torch.tensor(
            [
                [[9.375, 5.3], [3.8, 8], [3, 0.75]],
                [[6, 6], [10.125, 10.125], [1.875, 1.875]],
            ]
        )
        assert full_cat.plocs.equal(true_locs)

    def test_tile_cat_to_region_basic(self, basic_tilecat):
        region_cat = tile_cat_to_region_cat(basic_tilecat, 0.5, discard_extra_sources=False)
        full_cat = basic_tilecat.to_full_params()
        assert region_cat.to_full_catalog().plocs.equal(full_cat.plocs)

    def test_tile_cat_to_region_filtering(self):
        d = {
            "n_sources": torch.zeros(3, 2, 2),
            "locs": torch.zeros(3, 2, 2, 1, 2),
            "source_type": torch.ones((3, 2, 2, 1, 1)).bool(),
            "galaxy_params": torch.zeros((3, 2, 2, 1, 6)),
            "star_fluxes": torch.ones((3, 2, 2, 1, 5)) * 1000,
            "galaxy_fluxes": torch.ones(3, 2, 2, 1, 5) * 1000,
        }
        # BATCH 0: top right interior, center right boundary
        d["n_sources"][0, 0, 1] = 1
        d["n_sources"][0, 1, 1] = 1
        d["locs"][0, 0, 1, 0] = torch.tensor([0.5, 0.5])
        d["locs"][0, 1, 1, 0] = torch.tensor([0.02, 0.5])
        d["galaxy_fluxes"][0, 0, 1, 0, 2] = 5000  # keep top right

        # BATCH 1: top left interior, top center boundary
        d["n_sources"][1, 0, 0] = 1
        d["n_sources"][1, 0, 1] = 1
        d["locs"][1, 0, 0, 0] = torch.tensor([0.5, 0.5])
        d["locs"][1, 0, 1, 0] = torch.tensor([0.5, 0.02])
        d["galaxy_fluxes"][1, 0, 1, 0, 2] = 5000  # keep top center

        # BATCH 2: only one source in top left
        d["n_sources"][2, 0, 0] = 1
        d["locs"][2, 0, 0, 0] = torch.tensor([0.5, 0.5])

        tilecat = TileCatalog(4, d)

        # make sure no warning when converting (since extra sources have been discarded)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            region_cat = tile_cat_to_region_cat(tilecat, 0.5, discard_extra_sources=True)

        n_sources = region_cat.n_sources
        assert n_sources[0, 0, 2] == n_sources[1, 0, 1] == n_sources[2, 0, 0] == 1
        assert n_sources[0].sum() == n_sources[1].sum() == n_sources[2].sum()

    @pytest.fixture(scope="module")
    def region_cat(self):
        n_sources = torch.zeros(2, 5, 5)
        n_sources[0, 0, 0] = 1
        n_sources[0, 1, 3] = 1
        n_sources[0, 4, 2] = 1
        n_sources[1, 0, 0] = 1
        n_sources[1, 2, 2] = 1
        n_sources[1, 4, 4] = 1

        locs = torch.zeros(2, 5, 5, 1, 2)
        locs[0, 0, 0] = torch.tensor([0.8, 0.2])
        locs[0, 1, 3] = torch.tensor([0.1, 0.5])
        locs[0, 4, 2] = torch.tensor([0.3, 0.3])
        locs[1, 0, 0] = torch.tensor([0.5, 0.5])
        locs[1, 2, 2] = torch.tensor([0.5, 0.5])
        locs[1, 4, 4] = torch.tensor([0.5, 0.5])

        fluxes = torch.zeros(2, 5, 5, 1, 5)
        fluxes[0, 0, 0] = torch.tensor([100, 100, 100, 100, 100])
        fluxes[0, 1, 3] = torch.tensor([100, 100, 100, 100, 100]) * 2
        fluxes[0, 4, 2] = torch.tensor([100, 100, 100, 100, 100]) * 5

        d = {
            "n_sources": n_sources,
            "locs": locs,
            "galaxy_fluxes": fluxes,
        }

        return RegionCatalog(interior_slen=3.5, overlap_slen=0.5, d=d)
