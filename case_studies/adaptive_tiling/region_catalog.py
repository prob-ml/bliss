import itertools
import warnings
from collections import UserDict
from enum import IntEnum
from typing import Dict

import torch
from einops import rearrange, repeat
from torch import Tensor

from bliss.catalog import TileCatalog


class RegionType(IntEnum):
    INTERIOR = 0
    BOUNDARY_VERTICAL = 1
    BOUNDARY_HORIZONTAL = 2
    CORNER = 3


class RegionCatalog(TileCatalog, UserDict):
    # region Init
    def __init__(
        self,
        height: int = None,
        interior_slen: float = None,
        overlap_slen: float = None,
        d: Dict[str, Tensor] = None,
    ):
        super().__init__(None, d)

        self._validate_n_sources()
        assert self.max_sources == 1  # can only have one source per region
        msg = "Either height or interior_slen must be specified"
        assert height is not None or interior_slen is not None, msg
        assert overlap_slen is not None, "overlap_slen must be specified"

        self.n_rows = self.n_tiles_h
        self.n_cols = self.n_tiles_w
        self.nth = (self.n_rows + 1) // 2
        self.ntw = (self.n_rows + 1) // 2
        self.interior_slen = interior_slen if interior_slen else height / self.nth - overlap_slen
        self.overlap_slen = overlap_slen
        self.tile_slen = self.interior_slen + 2 * self.overlap_slen

        self.region_types = self._init_region_types()

    def _validate_n_sources(self):
        """Ensure that n_sources is valid for a region-based catalog.

        This validates both the shape and the constraint that there can be only one source per set
        of regions in a padded tile. We check this by moving a 3x3 window across the whole array
        and and checking that the sum of sources in that region is at most one.
        """
        _, n_h, n_w = self.n_sources.shape
        assert (n_h % 2 == 1) and (n_w % 2 == 1)  # need odd number of rows/cols
        # pad edges so each tile is 3x3, and stride by 2 to jump to next tile
        sources_per_tile = torch.nn.functional.unfold(
            self.n_sources.unsqueeze(1), kernel_size=(3, 3), padding=1, stride=2
        ).sum(axis=1)
        if not torch.all(sources_per_tile <= 1):
            warnings.warn("Some tiles contain more than one source.", stacklevel=2)

    def _init_region_types(self) -> Tensor:
        """Assign type to each region in n_rows x n_cols tensor (used for masking)."""
        mask = torch.zeros(self.n_rows, self.n_cols, device=self.device)
        mask[::2, ::2] = RegionType.INTERIOR
        mask[::2, 1::2] = RegionType.BOUNDARY_VERTICAL
        mask[1::2, ::2] = RegionType.BOUNDARY_HORIZONTAL
        mask[1::2, 1::2] = RegionType.CORNER
        return mask

    # endregion

    # region Properties
    @property
    def height(self) -> float:
        return self.nth * (self.interior_slen + self.overlap_slen)

    @property
    def width(self) -> float:
        return self.ntw * (self.interior_slen + self.overlap_slen)

    @property
    def is_on_mask(self) -> Tensor:
        # regions can only have one source, so we don't need to consider more than one
        return (self.n_sources == 1)[..., None]

    @property
    def interior_mask(self) -> Tensor:
        return self.region_types == RegionType.INTERIOR

    @property
    def vertical_boundary_mask(self) -> Tensor:
        return self.region_types == RegionType.BOUNDARY_VERTICAL

    @property
    def horizontal_boundary_mask(self) -> Tensor:
        return self.region_types == RegionType.BOUNDARY_HORIZONTAL

    @property
    def boundary_mask(self) -> Tensor:
        return self.vertical_boundary_mask | self.horizontal_boundary_mask

    @property
    def corner_mask(self) -> Tensor:
        return self.region_types == RegionType.CORNER

    # endregion

    # region Functions
    def to(self, device):
        out = {}
        for k, v in self.to_dict().items():
            out[k] = v.to(device)
        return RegionCatalog(
            interior_slen=self.interior_slen, overlap_slen=self.overlap_slen, d=out
        )

    def get_region_coords(self) -> Tensor:
        """Get pixel coordinates of top-left corner of regions."""
        x_coords = torch.arange(0, self.n_cols).float()
        y_coords = torch.arange(0, self.n_rows).float()
        coords = torch.cartesian_prod(y_coords, x_coords)

        bias = self.interior_slen * torch.floor_divide(coords + 1, 2)  # interior regions
        bias += self.overlap_slen * torch.floor_divide(coords, 2)  # overlap regions
        # account for half-overlap in first row/col
        bias[bias[..., 0] > 0] += torch.tensor([self.overlap_slen / 2, 0])
        bias[bias[..., 1] > 0] += torch.tensor([0, self.overlap_slen / 2])

        coords = rearrange(bias, "(nth ntw) d -> nth ntw d", nth=self.n_rows, ntw=self.n_cols)
        return coords.to(self.device)

    def get_region_sizes(self) -> Tensor:
        """Get sizes of regions in pixel coordinates."""
        sizes = torch.ones(self.n_rows, self.n_cols, 2) * self.interior_slen
        sizes[1::2, :, 0] = self.overlap_slen
        sizes[:, 1::2, 1] = self.overlap_slen
        # additional half-overlap in first and last row and col
        sizes[0, :, 0] += self.overlap_slen / 2
        sizes[-1, :, 0] += self.overlap_slen / 2
        sizes[:, 0, 1] += self.overlap_slen / 2
        sizes[:, -1, 1] += self.overlap_slen / 2
        return sizes.to(self.device)

    def get_tile_sizes(self) -> Tensor:
        """Get sizes of each tile."""
        tile_sizes = torch.ones(self.nth, self.ntw, 2, device=self.device) * self.tile_slen
        # handle half-overlap in first and last row and col of tiles, i.e. two regions
        tile_sizes[0, :, 0] -= self.overlap_slen / 2
        tile_sizes[-1, :, 0] -= self.overlap_slen / 2
        tile_sizes[:, 0, 1] -= self.overlap_slen / 2
        tile_sizes[:, -1, 1] -= self.overlap_slen / 2
        return tile_sizes

    def get_full_locs_from_regions(self) -> Tensor:
        """Convert locations in each region to locations in the full image.

        Returns:
            Tensor: b x nth x ntw x 1 x 2 tensor of locations in full coordinates
        """
        # get locs in pixel coords in each region
        plocs = self.locs * repeat(
            self.get_region_sizes(), "nth ntw d -> b nth ntw 1 d", b=self.batch_size
        )
        bias = repeat(
            self.get_region_coords(), "nth ntw d -> b nth ntw 1 d", b=self.batch_size
        ).float()
        return plocs + bias

    def get_full_locs_from_tiles(self) -> Tensor:
        """Here for compatibility with TileCatalog functions."""
        return self.get_full_locs_from_regions()

    def get_interior_locs_in_tile(self):
        """Get locations of interior regions relative to the tile its in."""
        b = self.batch_size
        locs = self.locs * repeat(self.get_region_sizes(), "nth ntw d -> b nth ntw 1 d", b=b)
        locs[:, 2::2, ::2, :, 0] += self.overlap_slen
        locs[:, ::2, 2::2, :, 1] += self.overlap_slen

        tile_sizes = repeat(self.get_tile_sizes(), "nth ntw d -> b nth ntw 1 d", b=b, d=2)
        tile_sizes = tile_sizes.reshape(-1).to(locs.dtype)
        mask = repeat(self.interior_mask, "nth ntw -> b nth ntw 1 d", b=b, d=2)
        locs[mask] /= tile_sizes
        return locs * mask

    def get_vertical_boundary_locs_in_tiles(self):
        """Get locations of boundary regions relative to tiles to left and right of the boundary."""
        b = self.batch_size
        locs = self.locs * repeat(self.get_region_sizes(), "nth ntw d -> b nth ntw 1 d", b=b)
        locs_i, locs_j = locs.clone(), locs.clone()
        tile_sizes = repeat(self.get_tile_sizes(), "nth ntw d -> b nth ntw 1 d", b=b, d=2)

        locs_i[:, 2::2, 1::2, :, 0] += self.overlap_slen
        locs_j[:, 2::2, 1::2, :, 0] += self.overlap_slen
        locs_i[:, :, 1, :, 1] += self.interior_slen + self.overlap_slen / 2  # half overlap
        locs_i[:, :, 3::2, :, 1] += self.interior_slen + self.overlap_slen  # full overlap

        mask = repeat(self.vertical_boundary_mask, "nth ntw -> b nth ntw 1 d", b=b, d=2)
        locs_i[mask] /= tile_sizes[:, :, :-1].reshape(-1).to(locs.dtype)
        locs_j[mask] /= tile_sizes[:, :, 1:].reshape(-1).to(locs.dtype)

        return locs_i * mask, locs_j * mask

    def get_horizontal_boundary_locs_in_tiles(self):
        """Get locations of boundary regions relative to tiles above and below the boundary."""
        b = self.batch_size
        locs = self.locs * repeat(self.get_region_sizes(), "nth ntw d -> b nth ntw 1 d", b=b)
        locs_i, locs_j = locs.clone(), locs.clone()
        tile_sizes = repeat(self.get_tile_sizes(), "nth ntw d -> b nth ntw 1 d", b=b, d=2)

        locs_i[:, 1::2, 2::2, :, 1] += self.overlap_slen
        locs_j[:, 1::2, 2::2, :, 1] += self.overlap_slen
        locs_i[:, 1, :, :, 0] += self.interior_slen + self.overlap_slen / 2  # half overlap
        locs_i[:, 3::2, :, :, 0] += self.interior_slen + self.overlap_slen  # full overlap

        mask = repeat(self.horizontal_boundary_mask, "nth ntw -> b nth ntw 1 d", b=b, d=2)
        locs_i[mask] /= tile_sizes[:, :-1].reshape(-1).to(locs.dtype)
        locs_j[mask] /= tile_sizes[:, 1:].reshape(-1).to(locs.dtype)

        return locs_i * mask, locs_j * mask

    def get_corner_locs_in_tiles(self):
        """Get locations of corner regions relative to tiles around it."""
        b = self.batch_size
        locs = self.locs * repeat(self.get_region_sizes(), "nth ntw d -> b nth ntw 1 d", b=b)
        locs = locs.clone().unsqueeze(0).repeat(4, 1, 1, 1, 1, 1)
        tile_sizes = repeat(self.get_tile_sizes(), "nth ntw d -> b nth ntw 1 d", b=b, d=2)

        locs[0:2, :, 1, :, :, 0] += self.interior_slen + self.overlap_slen / 2  # half overlap
        locs[0:2, :, 2:, :, :, 0] += self.interior_slen + self.overlap_slen  # full overlap
        locs[0::2, :, :, 1, :, 1] += self.interior_slen + self.overlap_slen / 2  # half overlap
        locs[0::2, :, :, 2:, :, 1] += self.interior_slen + self.overlap_slen  # full overlap

        mask = repeat(self.corner_mask, "nth ntw -> b nth ntw 1 d", b=b, d=2)
        locs[0][mask] /= tile_sizes[:, :-1, :-1].reshape(-1).to(locs.dtype)
        locs[1][mask] /= tile_sizes[:, :-1, 1:].reshape(-1).to(locs.dtype)
        locs[2][mask] /= tile_sizes[:, 1:, :-1].reshape(-1).to(locs.dtype)
        locs[3][mask] /= tile_sizes[:, 1:, 1:].reshape(-1).to(locs.dtype)

        return locs[0] * mask, locs[1] * mask, locs[2] * mask, locs[3] * mask

    def crop(self, hlims_tile, wlims_tile):
        d = {}
        for k, v in self.to_dict().items():
            d[k] = v[:, hlims_tile[0] : hlims_tile[1], wlims_tile[0] : wlims_tile[1]]
        return RegionCatalog(interior_slen=self.interior_slen, overlap_slen=self.overlap_slen, d=d)

    # endregion


def tile_cat_to_region_cat(tile_cat: TileCatalog, overlap_slen: float):
    """Convert a TileCatalog to RegionCatalog.

    We do this by checking if a location is within the interior or boundary, and copying the
    parameters to that region index in the new catalog. The locations are kept in full
    coordinates, and then rescaled to be between 0 and 1 within the region.

    Since we need to check each source individually, we use a for loop over each source in each
    tile in each batch in the original catalog.

    Args:
        tile_cat: the tile catalog to convert
        overlap_slen: the overlap in pixels between tiles

    Returns:
        RegionCatalog: the region-based representation of this TileCatalog
    """
    msg = "Only TileCatalogs with one source per tile can be converted to RegionCatalogs"
    assert tile_cat.max_sources == 1, msg
    n_rows = 2 * tile_cat.n_tiles_h - 1
    n_cols = 2 * tile_cat.n_tiles_w - 1

    d = {
        "locs": torch.zeros(tile_cat.batch_size, n_rows, n_cols, 1, 2),
        "n_sources": torch.zeros(tile_cat.batch_size, n_rows, n_cols),
    }
    for key, val in tile_cat.items():
        shape = list(val.shape)
        shape[1] = n_rows
        shape[2] = n_cols
        d[key] = torch.zeros(*shape)
    full_locs = tile_cat.get_full_locs_from_tiles()

    batch = list(range(tile_cat.batch_size))
    rows = list(range(tile_cat.n_tiles_h))
    cols = list(range(tile_cat.n_tiles_h))

    for b, i, j in itertools.product(batch, rows, cols):
        if tile_cat.n_sources[b, i, j] == 0:
            continue

        # Determine region to place source in based on
        threshold = (
            overlap_slen / 2 / tile_cat.tile_slen,
            1 - overlap_slen / 2 / tile_cat.tile_slen,
        )
        new_i, new_j = region_for_tile_source(  # noqa: WPS317
            tile_cat.locs[b, i, j], (i, j), n_rows, n_cols, threshold
        )

        d["locs"][b, new_i, new_j, 0] = full_locs[b, i, j, 0]
        d["n_sources"][b, new_i, new_j] = tile_cat.n_sources[b, i, j]
        for key, val in tile_cat.items():
            # initialize empty tensor
            if d.get(key) is None:
                d[key] = torch.zeros(
                    tile_cat.batch_size, n_rows, n_cols, tile_cat.max_sources, val.shape[-1]
                )
            d[key][b, new_i, new_j, 0] = val[b, i, j, 0]

    region_cat = RegionCatalog(height=tile_cat.height, overlap_slen=overlap_slen, d=d)

    offset = repeat(
        region_cat.get_region_coords(), "nth ntw d -> b nth ntw 1 d", b=tile_cat.batch_size
    )
    region_sizes = repeat(
        region_cat.get_region_sizes(), "nth ntw d -> b nth ntw 1 d", b=tile_cat.batch_size
    )
    region_cat.locs = ((region_cat.locs - offset) / region_sizes).clamp(0, 1)

    return region_cat


def region_for_tile_source(loc, pos, n_rows, n_cols, threshold):
    """Determine which region index a tile-based location should be placed in."""
    new_i, new_j = pos[0] * 2, pos[1] * 2

    if loc[0] < threshold[0] and new_i > 0:  # top edge
        new_i -= 1
    elif loc[0] > threshold[1] and new_i < n_rows - 1:  # bottom edge
        new_i += 1
    if loc[1] < threshold[0] and new_j > 0:  # left edge
        new_j -= 1
    elif loc[1] > threshold[1] and new_j < n_cols - 1:  # right edge
        new_j += 1
    return new_i, new_j
