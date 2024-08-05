import math
from collections import UserDict
from enum import IntEnum
from typing import Dict, Tuple

import torch
from astropy.wcs import WCS
from einops import rearrange, reduce, repeat
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def convert_mag_to_nmgy(mag):
    return 10 ** ((22.5 - mag) / 2.5)


def convert_nmgy_to_mag(nmgy):
    return 22.5 - 2.5 * torch.log10(nmgy)


def convert_nmgy_to_njymag(nmgy):
    """Convert from flux (nano-maggie) to mag (nano-jansky), which is the format used by DC2.

    For the difference between mag (Pogson magnitude) and njymag (AB magnitude), please view
    the "Flux units: maggies and nanomaggies" part of
    https://www.sdss3.org/dr8/algorithms/magnitudes.php#nmgy
    When we change the standard source to AB sources, we need to do the conversion
    described in "2.10 AB magnitudes" at
    https://pstn-001.lsst.io/fluxunits.pdf

    Args:
        nmgy: the fluxes in nanomaggies

    Returns:
        Tensor indicating fluxes in AB magnitude
    """

    return 22.5 - 2.5 * torch.log10(nmgy / 3631)


class SourceType(IntEnum):
    STAR = 0
    GALAXY = 1


class BaseTileCatalog(UserDict):
    def __init__(self, d: Dict[str, Tensor]):
        v = next(iter(d.values()))
        self.batch_size, self.n_tiles_h, self.n_tiles_w = v.shape[:3]
        self.device = v.device

        super().__init__(**d)

    def __setitem__(self, key: str, item: Tensor) -> None:
        self._validate(item)
        # TODO: all float data should be torch.float32, fix this
        if item.dtype == torch.float64:
            item = item.float()
        super().__setitem__(key, item)

    def _validate(self, x: Tensor):
        assert isinstance(x, Tensor)
        assert x.shape[:3] == (self.batch_size, self.n_tiles_h, self.n_tiles_w)
        assert x.device == self.device

    def to(self, device):
        out = {}
        for k, v in self.items():
            out[k] = v.to(device)
        return type(self)(out)

    def crop(self, hlims_tile, wlims_tile):
        out = {}
        for k, v in self.items():
            out[k] = v[:, hlims_tile[0] : hlims_tile[1], wlims_tile[0] : wlims_tile[1]]
        return type(self)(out)

    def symmetric_crop(self, tiles_to_crop):
        return self.crop(
            [tiles_to_crop, self.n_tiles_h - tiles_to_crop],
            [tiles_to_crop, self.n_tiles_w - tiles_to_crop],
        )

    def filter_base_tile_catalog_by_ploc_box(self, box_origin: torch.Tensor, box_len: float):
        assert box_origin[0] + box_len < self.height, "invalid box"
        assert box_origin[1] + box_len < self.width, "invalid box"

        box_origin_tensor = box_origin.view(1, 1, 2).to(device=self.device)
        box_end_tensor = (box_origin + box_len).view(1, 1, 2).to(device=self.device)

        plocs_mask = torch.all(
            (self["plocs"] < box_end_tensor) & (self["plocs"] > box_origin_tensor), dim=2
        )

        plocs_mask_indexes = plocs_mask.nonzero()
        plocs_inverse_mask_indexes = (~plocs_mask).nonzero()
        plocs_full_mask_indexes = torch.cat((plocs_mask_indexes, plocs_inverse_mask_indexes), dim=0)
        _, index_order = plocs_full_mask_indexes[:, 0].sort(stable=True)
        plocs_full_mask_sorted_indexes = plocs_full_mask_indexes[index_order.tolist(), :]

        d = {}
        new_max_sources = plocs_mask.sum(dim=1).max()
        for k, v in self.items():
            if k == "n_sources":
                d[k] = plocs_mask.sum(dim=1)
            else:
                d[k] = v[
                    plocs_full_mask_sorted_indexes[:, 0].tolist(),
                    plocs_full_mask_sorted_indexes[:, 1].tolist(),
                ].view(-1, self.max_sources, v.shape[-1])[:, :new_max_sources, :]

        d["plocs"] -= box_origin_tensor

        return FullCatalog(box_len, box_len, d)


class TileCatalog(BaseTileCatalog):
    galaxy_params = [
        "galaxy_disk_frac",
        "galaxy_beta_radians",
        "galaxy_disk_q",
        "galaxy_a_d",
        "galaxy_bulge_q",
        "galaxy_a_b",
    ]
    galaxy_params_index = {k: i for i, k in enumerate(galaxy_params)}

    def __init__(self, d: Dict[str, Tensor]):
        self.max_sources = d["locs"].shape[3]
        super().__init__(d)

    def __getitem__(self, name: str):
        # a temporary hack until we stop storing galaxy_params as an array
        # TODO: remove this hack
        if "galaxy_params" in self.keys() and name in self.galaxy_params:
            idx = self.galaxy_params_index[name]
            return self.data["galaxy_params"][..., idx : (idx + 1)]
        return super().__getitem__(name)

    @property
    def is_on_mask(self) -> Tensor:
        """Provides tensor which indicates how many sources are present for each batch.

        Return a boolean array of `shape=(*n_sources.shape, max_sources)` whose `(*,l)th` entry
        indicates whether there are more than l sources on the `*th` index.

        Returns:
            Tensor indicating how many sources are present for each batch.
        """
        # TODO: a tile catalog should store the is_on_mask explicitly, not derive it from n_sources.
        # perhaps we should have some catalog format that can store at most one source per tile
        arange = torch.arange(self.max_sources, device=self.device)
        arange = arange.expand(*self["n_sources"].shape, self.max_sources)
        return arange < self["n_sources"].unsqueeze(-1)

    @property
    def star_bools(self) -> Tensor:
        is_star = self["source_type"] == SourceType.STAR
        return is_star * self.is_on_mask.unsqueeze(-1)

    @property
    def galaxy_bools(self) -> Tensor:
        is_galaxy = self["source_type"] == SourceType.GALAXY
        return is_galaxy * self.is_on_mask.unsqueeze(-1)

    def on_fluxes(self, unit: str):
        match unit:
            case "nmgy":
                return self.on_nmgy
            case "mag":
                return self.on_mag
            case "njymag":
                return self.on_njymag
            case _:
                raise NotImplementedError()

    @property
    def on_nmgy(self):
        # TODO: a tile catalog should store fluxes rather than star_fluxes and galaxy_fluxes
        # because that's all that's needed to render the source
        if "galaxy_fluxes" not in self:
            fluxes = self["star_fluxes"]
        else:
            fluxes = torch.where(self.galaxy_bools, self["galaxy_fluxes"], self["star_fluxes"])
        return torch.where(self.is_on_mask[..., None], fluxes, torch.zeros_like(fluxes))

    @property
    def on_mag(self) -> Tensor:
        return convert_nmgy_to_mag(self.on_nmgy)

    @property
    def on_njymag(self) -> Tensor:
        return convert_nmgy_to_njymag(self.on_nmgy)

    def to_full_catalog(self, tile_slen):
        """Converts image parameters in tiles to parameters of full image.

        Args:
            tile_slen: The side length of the square tiles (in pixels).

        Returns:
            The FullCatalog instance corresponding to the TileCatalog instance.

            NOTE: The locations (`"locs"`) are between 0 and 1. The output also contains
            pixel locations ("plocs") that are between 0 and `slen`.
        """
        plocs = self._get_plocs_from_tiles(tile_slen)
        param_names_to_mask = {"plocs"}.union(set(self.keys()))
        tile_params_to_gather = {"plocs": plocs}
        tile_params_to_gather.update(self)

        params = {}
        indices_to_retrieve, is_on_array = self.get_indices_of_on_sources()
        for param_name, tile_param in tile_params_to_gather.items():
            if param_name == "n_sources":
                continue
            if param_name == "locs":  # full catalog uses plocs instead of locs
                continue
            k = tile_param.shape[-1]
            param = rearrange(tile_param, "b nth ntw s k -> b (nth ntw s) k", k=k)
            indices_for_param = repeat(indices_to_retrieve, "b nth_ntw_s -> b nth_ntw_s k", k=k)
            param = torch.gather(param, dim=1, index=indices_for_param)
            if param_name in param_names_to_mask:
                param = param * is_on_array.unsqueeze(-1)
            params[param_name] = param

        params["n_sources"] = reduce(self["n_sources"], "b nth ntw -> b", "sum")

        height_px = self.n_tiles_h * tile_slen
        width_px = self.n_tiles_w * tile_slen

        return FullCatalog(height_px, width_px, params)

    def _get_plocs_from_tiles(self, tile_slen) -> Tensor:
        """Get the full image locations from tile locations.

        Args:
            tile_slen: The side length of the square tiles (in pixels).

        Returns:
            Tensor: pixel coordinates of each source (between 0 and slen).
        """
        slen = self.n_tiles_h * tile_slen
        wlen = self.n_tiles_w * tile_slen
        # coordinates on tiles.
        x_coords = torch.arange(0, slen, tile_slen, device=self["locs"].device).long()
        y_coords = torch.arange(0, wlen, tile_slen, device=self["locs"].device).long()
        tile_coords = torch.cartesian_prod(x_coords, y_coords)

        # recenter and renormalize locations.
        locs = rearrange(self["locs"], "b nth ntw d xy -> (b nth ntw) d xy", xy=2)
        bias = repeat(tile_coords, "n xy -> (r n) 1 xy", r=self.batch_size).float()

        plocs = locs * tile_slen + bias
        return rearrange(
            plocs,
            "(b nth ntw) d xy -> b nth ntw d xy",
            b=self.batch_size,
            nth=self.n_tiles_h,
            ntw=self.n_tiles_w,
        )

    def get_indices_of_on_sources(self) -> Tuple[Tensor, Tensor]:
        """Get the indices of detected sources from each tile.

        Returns:
            A 2-D tensor of integers with shape `n_samples x max(n_sources)`,
            where `max(n_sources)` is the maximum number of sources detected across all samples.

            Each element of this tensor is an index of a particular source within a particular tile.
            For a particular sample i that had N detections,
            the first N indices of indices_sorted[i. :] will be the detected sources.
            This is accomplished by flattening the n_tiles_per_image and max_detections.
            For example, if we had 3 tiles with a maximum of two sources each,
            the elements of this tensor would take values from 0 up to and including 5.
        """
        tile_is_on_array_sampled = self.is_on_mask
        n_sources = reduce(tile_is_on_array_sampled, "b nth ntw d -> b", "sum")
        max_sources = int(n_sources.max().int().item())
        tile_is_on_array = rearrange(tile_is_on_array_sampled, "b nth ntw d -> b (nth ntw d)")
        indices_sorted = tile_is_on_array.long().argsort(dim=1, descending=True)
        indices_sorted = indices_sorted[:, :max_sources]

        is_on_array = torch.gather(tile_is_on_array, dim=1, index=indices_sorted)
        return indices_sorted, is_on_array

    def _sort_sources_by_flux(self, band=2):
        # sort by fluxes of "on" sources to get brightest source per tile
        on_nmgy = self.on_nmgy[..., band]  # shape n x nth x ntw x d
        top_indexes = on_nmgy.argsort(dim=3, descending=True)

        d = {"n_sources": self["n_sources"]}
        for key, val in self.items():
            if key != "n_sources":
                param_dim = val.size(-1)
                idx_to_gather = repeat(top_indexes, "... -> ... pd", pd=param_dim)
                d[key] = torch.take_along_dim(val, idx_to_gather, dim=3)

        return TileCatalog(d)

    def get_brightest_sources_per_tile(self, top_k=1, exclude_num=0, band=2):
        """Restrict TileCatalog to only the brightest 'on' source per tile.

        Args:
            top_k (int): The number of sources to keep per tile. Defaults to 1.
            exclude_num (int): A number of the brightest sources to exclude. Defaults to 0.
            band (int): The band to compare fluxes in. Defaults to 2 (r-band).

        Returns:
            TileCatalog: a new catalog with only one source per tile
        """
        if self.max_sources == 1 and top_k == 1 and exclude_num == 0:
            return self

        if exclude_num >= self.max_sources:
            tc = TileCatalog(self.data)
            tc["n_sources"] = torch.zeros_like(tc["n_sources"])
            return tc

        sorted_self = self._sort_sources_by_flux(band=band)

        d = {}
        for key, val in sorted_self.items():
            if key == "n_sources":
                d[key] = (sorted_self["n_sources"] - exclude_num).clamp(min=0, max=top_k)
            else:
                d[key] = val[:, :, :, exclude_num : (exclude_num + top_k)]

        return TileCatalog(d)

    def filter_by_flux(self, min_flux=0, max_flux=torch.inf, band=2):
        """Restricts TileCatalog to sources that have a flux between min_flux and max_flux.

        Args:
            min_flux (float): Minimum flux value to keep. Defaults to 0.
            max_flux (float): Maximum flux value to keep. Defaults to infinity.
            band (int): The band to compare fluxes in. Defaults to 2 (r-band).

        Returns:
            TileCatalog: a new catalog with only sources within the flux range. Note that the size
                of the tensors stays the same, but params at sources outside the range are set to 0.
        """
        sorted_self = self._sort_sources_by_flux(band=band)

        # get fluxes of "on" sources to mask by
        on_nmgy = sorted_self.on_nmgy[..., band]
        flux_mask = (on_nmgy > min_flux) & (on_nmgy < max_flux)

        d = {}
        for key, val in sorted_self.items():
            if key == "n_sources":
                d[key] = flux_mask.sum(dim=3)  # number of sources within range in tile
            else:
                d[key] = torch.where(flux_mask.unsqueeze(-1), val, torch.zeros_like(val))

        return TileCatalog(d)

    def union(self, other, disjoint=False):
        """Returns a new TileCatalog containing the union of the sources in self and other.
        The maximum number of sources in the returned catalog is the sum of the maximum number
        of sources in self and other if disjoint is false, otherwise it is unchanged.

        Args:
            other: Another TileCatalog.
            disjoint: whether the catalogs cannot have sources in the same tiles

        Returns:
            A new TileCatalog containing the union of the sources in self and other.
        """
        assert self.batch_size == other.batch_size
        assert self.n_tiles_h == other.n_tiles_h
        assert self.n_tiles_w == other.n_tiles_w

        d = {}
        ns11 = rearrange(self["n_sources"], "b ht wt -> b ht wt 1 1")
        for k, v in self.items():
            if k == "n_sources":
                d[k] = v + other[k]
                if disjoint:
                    assert d[k].max() <= 1
            else:
                if disjoint:
                    d1 = v
                    d2 = other[k]
                else:
                    d1 = torch.cat((v, other[k]), dim=-2)
                    d2 = torch.cat((other[k], v), dim=-2)
                d[k] = torch.where(ns11 > 0, d1, d2)
        return TileCatalog(d)

    def __repr__(self):
        keys = ", ".join(self.keys())
        return f"TileCatalog({self.batch_size} x {self.n_tiles_h} x {self.n_tiles_w}; {keys})"


class FullCatalog(UserDict):
    @staticmethod
    def plocs_from_ra_dec(ras, decs, wcs: WCS):
        """Converts RA/DEC coordinates into BLISS's pixel coordinates.
            BLISS pixel coordinates have (0, 0) as the lower-left corner, whereas standard pixel
            coordinates begin at (-0.5, -0.5). BLISS pixel coordinates are in row-column order,
            whereas standard pixel coordinates are in column-row order.

        Args:
            ras (Tensor): (b, n) tensor of RA coordinates in degrees.
            decs (Tensor): (b, n) tensor of DEC coordinates in degrees.
            wcs (WCS): WCS object to use for transformation.

        Returns:
            A 1xNx2 tensor containing the locations of the light sources in pixel coordinates. This
            function does not write self["plocs"], so you should do that manually if necessary.
        """
        ras = ras.numpy().squeeze()
        decs = decs.numpy().squeeze()

        pt, pr = wcs.all_world2pix(ras, decs, 0)  # convert to pixel coordinates
        pt = torch.tensor(pt) + 0.5  # For consistency with BLISS
        pr = torch.tensor(pr) + 0.5
        return torch.stack((pr, pt), dim=-1)

    @classmethod
    def from_file(cls, cat_path, wcs, height, width, **kwargs) -> "FullCatalog":
        """Loads FullCatalog from disk."""
        raise NotImplementedError

    def __init__(self, height: int, width: int, d: Dict[str, Tensor]) -> None:
        """Initialize FullCatalog.

        Args:
            height: In pixels, without accounting for border padding.
            width: In pixels, without accounting for border padding.
            d: Dictionary containing parameters of FullCatalog with correct shape.
        """
        self.height = height
        self.width = width

        self.device = d["plocs"].device
        self.batch_size, self.max_sources, hw = d["plocs"].shape
        assert hw == 2
        assert d["n_sources"].max().int().item() <= self.max_sources
        assert d["n_sources"].shape == (self.batch_size,)

        super().__init__(**d)

    def __setitem__(self, key: str, item: Tensor) -> None:
        self._validate(item)
        super().__setitem__(key, item)

    def _validate(self, x: Tensor):
        assert isinstance(x, Tensor)
        assert x.shape[0] == self.batch_size
        assert x.device == self.device

    def to(self, device):
        out = {}
        for k, v in self.items():
            out[k] = v.to(device)
        return type(self)(self.height, self.width, out)

    @property
    def is_on_mask(self) -> Tensor:
        arange = torch.arange(self.max_sources, device=self.device)
        return arange.view(1, -1) < self["n_sources"].view(-1, 1)

    @property
    def star_bools(self) -> Tensor:
        is_star = self["source_type"] == SourceType.STAR
        assert is_star.size(1) == self.max_sources
        assert is_star.size(2) == 1
        return is_star * self.is_on_mask.unsqueeze(2)

    @property
    def galaxy_bools(self) -> Tensor:
        is_galaxy = self["source_type"] == SourceType.GALAXY
        assert is_galaxy.size(1) == self.max_sources
        assert is_galaxy.size(2) == 1
        return is_galaxy * self.is_on_mask.unsqueeze(2)

    def on_fluxes(self, unit: str):
        match unit:
            case "nmgy":
                return self.on_nmgy
            case "mag":
                return self.on_mag
            case "njymag":
                return self.on_njymag
            case _:
                raise NotImplementedError()

    @property
    def on_nmgy(self) -> Tensor:
        # ideally we'd always store fluxes rather than star_fluxes and galaxy_fluxes
        if "galaxy_fluxes" not in self:
            fluxes = self["star_fluxes"]
        else:
            fluxes = torch.where(self.galaxy_bools, self["galaxy_fluxes"], self["star_fluxes"])
        return torch.where(self.is_on_mask[..., None], fluxes, torch.zeros_like(fluxes))

    @property
    def on_mag(self) -> Tensor:
        return convert_nmgy_to_mag(self.on_nmgy)

    @property
    def on_njymag(self) -> Tensor:
        return convert_nmgy_to_njymag(self.on_nmgy)

    def one_source(self, b: int, s: int):
        """Return a dict containing all parameter for one specified light source."""
        out = {}
        for k, v in self.items():
            if k == "n_sources":
                assert s < v[b]
                continue
            out[k] = v[b][s]
        return out

    def apply_param_bin(self, pname: str, p_min: float, p_max: float):
        """Apply magnitude bin to given parameters."""
        assert pname in self, f"Parameter '{pname}' required to apply mag cut."
        assert self[pname].shape[-1] == 1, "Can only be applied to scalar parameters."
        assert self[pname].min().item() >= 0, f"Cannot use this function with {pname}."
        assert p_min >= 0, "`p_min` should be at least 0 "

        # get indices to collect
        keep = self[pname] < p_max
        keep = keep & (self[pname] > p_min)
        n_batches, max_sources, _ = keep.shape
        as_indices = torch.arange(0, max_sources).expand(n_batches, max_sources).unsqueeze(-1)
        to_collect = torch.where(keep, as_indices, torch.ones_like(keep) * max_sources)
        to_collect = to_collect.sort(dim=1)[0]

        # get dictionary with all params (including plocs)
        d = dict(self.items())
        d_new = {}
        d["plocs"] = self["plocs"]
        for k, v in d.items():
            if k == "n_sources":
                continue
            pdim = v.shape[-1]
            to_collect_v = to_collect.expand(n_batches, max_sources, pdim)
            v_expand = torch.hstack([v, torch.zeros(v.shape[0], 1, v.shape[-1])])
            d_new[k] = torch.gather(v_expand, 1, to_collect_v)
        d_new["n_sources"] = keep.sum(dim=(-2, -1)).long()
        return type(self)(self.height, self.width, d_new)

    def _get_tile_to_source_mapping(self, mapping_matrix: Tensor) -> Tensor:
        return mapping_matrix.nonzero()

    def to_tile_catalog(
        self,
        tile_slen: int,
        max_sources_per_tile: int,
        ignore_extra_sources=False,
        filter_oob=False,
    ) -> TileCatalog:
        """Returns the TileCatalog corresponding to this FullCatalog.

        Args:
            tile_slen: The side length of the tiles.
            max_sources_per_tile: The maximum number of sources in one tile.
            ignore_extra_sources: If False (default), raises an error if the number of sources
                in one tile exceeds the `max_sources_per_tile`. If True, only adds the tile
                parameters of the first `max_sources_per_tile` sources to the new TileCatalog.
            filter_oob: If filter_oob is true, filter out the sources outside the image. (e.g. In
                case of data augmentation, there is a chance of some sources located outside the
                image)

        Returns:
            TileCatalog correspond to the each source in the FullCatalog.

        Raises:
            ValueError: If the number of sources in one tile exceeds `max_sources_per_tile` and
                `ignore_extra_sources` is False.
            KeyError: If the tile_params contain `plocs` or `n_sources`.
        """
        # TODO: a FullCatalog only needs to "know" its height and width to convert itself to a
        # TileCatalog. So those parameters should be passed on conversion, not initialization.
        tile_coords = torch.div(self["plocs"], tile_slen, rounding_mode="trunc").to(torch.int)
        n_tiles_h = math.ceil(self.height / tile_slen)
        n_tiles_w = math.ceil(self.width / tile_slen)

        # prepare tiled tensors
        tile_cat_shape = (self.batch_size, n_tiles_h, n_tiles_w, max_sources_per_tile)
        tile_locs = torch.zeros((*tile_cat_shape, 2), device=self.device)
        tile_n_sources = torch.zeros(tile_cat_shape[:3], dtype=torch.int64, device=self.device)
        tile_params: Dict[str, Tensor] = {}
        for k, v in self.items():
            if k in {"plocs", "n_sources"}:
                continue
            size = (self.batch_size, n_tiles_h, n_tiles_w, max_sources_per_tile, v.shape[-1])
            tile_params[k] = torch.zeros(size, dtype=v.dtype, device=self.device)

        tile_params["locs"] = tile_locs

        for ii in range(self.batch_size):
            n_sources = int(self["n_sources"][ii].item())
            plocs_ii = self["plocs"][ii][:n_sources]
            filter_sources = n_sources
            source_tile_coords = tile_coords[ii][:n_sources]
            if filter_oob:
                x0_mask = (plocs_ii[:, 0] > 0) & (plocs_ii[:, 0] < self.height)
                x1_mask = (plocs_ii[:, 1] > 0) & (plocs_ii[:, 1] < self.width)
                x_mask = x0_mask * x1_mask
                filter_sources = x_mask.sum()
                source_tile_coords = source_tile_coords[x_mask]

            if filter_sources == 0:
                continue

            source_indices = source_tile_coords[:, 0] * n_tiles_w + source_tile_coords[
                :, 1
            ].unsqueeze(0)
            tile_indices = torch.arange(n_tiles_h * n_tiles_w, device=self.device).unsqueeze(1)

            tile_to_source_mapping = self._get_tile_to_source_mapping(
                source_indices == tile_indices
            )
            tile_source_count: Tuple[Tensor, Tensor] = tile_to_source_mapping[:, 0].unique(
                sorted=True, return_counts=True
            )  # first element is tile index; second element is source count
            if tile_source_count[1].max() > max_sources_per_tile:
                if not ignore_extra_sources:
                    raise ValueError(  # noqa: WPS220
                        "# of sources per tile exceeds `max_sources_per_tile`."
                    )

            # get n_sources for each tile
            tile_n_sources[ii].view(-1)[tile_source_count[0].flatten().tolist()] = torch.where(
                tile_source_count[1] <= max_sources_per_tile,
                tile_source_count[1],
                max_sources_per_tile,
            )

            for k, v in tile_params.items():
                if k == "plocs":
                    raise KeyError("plocs should not be in tile_params")
                if k == "locs":
                    k = "plocs"
                if k == "n_sources":
                    raise KeyError("n_sources should not be in tile_params")
                param_matrix = self[k][ii][:n_sources]
                if filter_oob:
                    param_matrix = param_matrix[x_mask]
                params_on_tile = list(
                    param_matrix[tile_to_source_mapping[:, 1]].split(
                        tile_source_count[1].flatten().tolist()
                    )
                )
                # pad first tensor to desired length
                # the second argument of pad function is
                # padding_left, padding_right, padding_top, padding_bottom
                params_on_tile[0] = torch.nn.functional.pad(
                    params_on_tile[0],
                    (0, 0, 0, (max_sources_per_tile - params_on_tile[0].shape[0])),
                )
                # pad all tensors
                params_on_tile = pad_sequence(params_on_tile, batch_first=True)
                max_fill = min(filter_sources, max_sources_per_tile)
                v[ii].view(-1, *v[ii].shape[2:])[
                    tile_to_source_mapping[:, 0].unique(sorted=True).tolist(), :max_fill
                ] = params_on_tile[:, :max_fill].to(dtype=v.dtype)

            # modify tile location
            tile_params["locs"][ii] = (tile_params["locs"][ii] % tile_slen) / tile_slen
        tile_params.update({"n_sources": tile_n_sources})
        return TileCatalog(tile_params)

    def filter_by_ploc_box(self, box_origin: torch.Tensor, box_len: float):
        assert box_origin[0] + box_len <= self.height, "invalid box"
        assert box_origin[1] + box_len <= self.width, "invalid box"

        box_origin_tensor = box_origin.view(1, 1, 2).to(device=self.device)
        box_end_tensor = (box_origin + box_len).view(1, 1, 2).to(device=self.device)

        plocs_mask = torch.all(
            (self["plocs"] < box_end_tensor) & (self["plocs"] > box_origin_tensor), dim=2
        )

        plocs_mask_indexes = plocs_mask.nonzero()
        plocs_inverse_mask_indexes = (~plocs_mask).nonzero()
        plocs_full_mask_indexes = torch.cat((plocs_mask_indexes, plocs_inverse_mask_indexes), dim=0)
        _, index_order = plocs_full_mask_indexes[:, 0].sort(stable=True)
        plocs_full_mask_sorted_indexes = plocs_full_mask_indexes[index_order.tolist(), :]

        d = {}
        new_max_sources = plocs_mask.sum(dim=1).max()
        for k, v in self.items():
            if k == "n_sources":
                d[k] = plocs_mask.sum(dim=1)
            else:
                d[k] = v[
                    plocs_full_mask_sorted_indexes[:, 0].tolist(),
                    plocs_full_mask_sorted_indexes[:, 1].tolist(),
                ].view(-1, self.max_sources, v.shape[-1])[:, :new_max_sources, :]

        d["plocs"] -= box_origin_tensor

        return FullCatalog(box_len, box_len, d)
