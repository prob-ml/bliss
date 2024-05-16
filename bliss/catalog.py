import math
from collections import UserDict
from copy import copy
from typing import Dict, List, Tuple

import torch
from einops import rearrange, reduce, repeat
from tensordict import TensorDict
from torch import Tensor


class TileCatalog(UserDict):
    allowed_params = {
        "galaxy_params",
        "galaxy_bools",
        "fluxes",
        "mags",
        "locs_sd",
        "ellips",
        "snr",
        "blendedness",
        "galaxy_fluxes",
        "galaxy_probs",
        "star_fluxes",
        "star_log_fluxes",
        "star_bools",
        "n_source_probs",
    }

    def __init__(self, tile_slen: int, d: Dict[str, Tensor]):
        self.tile_slen = tile_slen
        d = copy(d)  # shallow copy, so we don't mutate the argument
        self.locs = d.pop("locs")
        self.n_sources = d.pop("n_sources").long()

        bs, nth, ntw, _ = self.locs.shape
        assert self.n_sources.max() <= 1 and self.n_sources.min() >= 0
        assert self.locs.max() <= 1 and self.locs.min() >= 0
        assert self.n_sources.shape == (bs, nth, ntw)
        self.batch_size = bs
        self.nth = nth
        self.ntw = ntw

        super().__init__(**d)

    def __getitem__(self, key: str) -> Tensor:
        assert isinstance(key, str)
        return super().__getitem__(key)

    def __setitem__(self, key: str, item: Tensor) -> None:
        if key not in self.allowed_params:
            raise ValueError(f"The key '{key}' is not in the allowed parameters for TileCatalog")
        self._validate(item)
        super().__setitem__(key, item)

    @classmethod
    def from_flat_dict(cls, tile_slen: int, nth: int, ntw: int, d: Dict[str, Tensor]):
        catalog_dict: Dict[str, Tensor] = {}
        catalog_dict["n_sources"] = rearrange(
            d["n_sources"], "(b nth ntw) -> b nth ntw", nth=nth, ntw=ntw
        )
        for k, v in d.items():
            if k != "n_sources":
                catalog_dict[k] = rearrange(v, "(b nth ntw) d -> b nth ntw d", nth=nth, ntw=ntw)
        return cls(tile_slen, catalog_dict)

    def cpu(self):
        return self.to("cpu")

    def to(self, device):
        out = {}
        for k, v in self.to_dict().items():
            out[k] = v.to(device)
        return type(self)(self.tile_slen, out)

    def to_dict(self) -> Dict[str, Tensor]:
        out = {}
        out["locs"] = self.locs
        out["n_sources"] = self.n_sources
        for k, v in self.items():
            out[k] = v
        return out

    @property
    def device(self):
        return self.locs.device

    def to_full_params(self) -> "FullCatalog":
        """Converts image parameters in tiles to parameters of full image.

        By parameters, we mean samples from the variational distribution, not the variational
        parameters.

        Returns:
            The FullCatalog instance corresponding to the TileCatalog instance.

            NOTE: The locations (`"locs"`) are between 0 and 1. The output also contains
            pixel locations ("plocs") that are between 0 and `slen`.
        """
        plocs = _get_tiled_plocs(self.locs, self.tile_slen)

        # include plocs with other parameters (excluding `n_sources` and `locs`)
        param_names_to_mask = {"plocs"}.union(set(self.keys()))
        tile_params_to_gather = {"plocs": plocs}
        tile_params_to_gather.update(self)

        params = {}
        indices_to_retrieve, is_on_array = self._get_indices_of_on_sources()
        for param_name, tile_param in tile_params_to_gather.items():
            k = tile_param.shape[-1]
            param = rearrange(tile_param, "b nth ntw k -> b (nth ntw) k", k=k)
            indices_for_param = repeat(indices_to_retrieve, "b nth_ntw -> b nth_ntw k", k=k)
            param = torch.gather(param, dim=1, index=indices_for_param)
            if param_name in param_names_to_mask:
                param = param * rearrange(is_on_array, "b (nth_ntw 1) -> b nth_ntw 1")
            params[param_name] = param

        params["n_sources"] = reduce(self.n_sources, "b nth ntw -> b", "sum")
        height, width = self.nth * self.tile_slen, self.ntw * self.tile_slen
        return FullCatalog(height, width, params)

    def get_tile_params_at_coord(self, plocs: torch.Tensor) -> Dict[str, Tensor]:
        """Return the parameters contained within the tiles corresponding to locations in plocs."""
        assert len(plocs.shape) == 2 and plocs.shape[1] == 2
        assert plocs.device == self.locs.device
        n_total = len(plocs)
        slen = self.nth * self.tile_slen
        wlen = self.ntw * self.tile_slen

        # coordinates on tiles.
        x_coords = torch.arange(0, slen, self.tile_slen, device=self.locs.device).long()
        y_coords = torch.arange(0, wlen, self.tile_slen, device=self.locs.device).long()

        x_indx = torch.searchsorted(x_coords.contiguous(), plocs[:, 0].contiguous()) - 1
        y_indx = torch.searchsorted(y_coords.contiguous(), plocs[:, 1].contiguous()) - 1

        # gather in dictionary
        d = {k: v[:, x_indx, y_indx, :].reshape(n_total, -1) for k, v in self.items()}

        # also include locs
        d["locs"] = self.locs[:, x_indx, y_indx, :].reshape(n_total, -1)

        return d

    def _get_indices_of_on_sources(self) -> Tuple[Tensor, Tensor]:
        """Get the indices of detected sources from each tile.

        Returns:
            A 2-D tensor of integers with shape `n_samples x max(n_sources)`,
            where `max(n_sources)` is the maximum number of sources detected across all samples.

            Each element of this tensor is an index of a particular source within a particular tile.
            For a particular sample i that had N detections,
            the first N indices of indices_sorted[i. :] will be the detected sources.
            This is accomplishied by flattening the n_tiles_per_image and max_detections.
            For example, if we had 3 tiles with a maximum of two sources each,
            the elements of this tensor would take values from 0 up to and including 5.
        """
        n_sources_per_batch = reduce(self.n_sources, "b nth ntw -> b", "sum")
        max_n_sources_per_batch = n_sources_per_batch.max().int()
        tile_is_on_array = rearrange(self.n_sources, "b nth ntw-> b (nth ntw)")
        indices_sorted = tile_is_on_array.long().argsort(dim=1, descending=True)
        indices_sorted_clipped = indices_sorted[:, :max_n_sources_per_batch]

        is_on_array = torch.gather(tile_is_on_array, dim=1, index=indices_sorted_clipped)
        return indices_sorted_clipped, is_on_array

    def _validate(self, x: Tensor):
        assert isinstance(x, Tensor)
        assert x.ndim == 4
        assert x.shape[:-1] == (self.batch_size, self.nth, self.ntw)
        assert x.device == self.device


def _get_tiled_plocs(locs: Tensor, tile_slen: int) -> Tensor:
    """Get the full image locations from tile locations.

    Returns:
        "plocs" are the pixel coordinates of each source (between 0 and slen).
    """
    batch_size, nth, ntw, _ = locs.shape
    slen = nth * tile_slen
    wlen = ntw * tile_slen

    # coordinates on tiles.
    x_coords = torch.arange(0, slen, tile_slen, device=locs.device).long()
    y_coords = torch.arange(0, wlen, tile_slen, device=locs.device).long()
    tile_coords = torch.cartesian_prod(x_coords, y_coords)

    # recenter and renormalize locations.
    locs_flat = rearrange(locs, "b nth ntw xy -> (b nth ntw) xy", xy=2)
    bias = repeat(tile_coords, "n xy -> (r n) xy", r=batch_size).float()

    plocs = locs_flat * tile_slen + bias
    return rearrange(plocs, "(b nth ntw) xy -> b nth ntw xy", b=batch_size, nth=nth, ntw=ntw)


class FullCatalog(UserDict):
    allowed_params = TileCatalog.allowed_params

    def __init__(self, height: int, width: int, d: dict[str, Tensor]) -> None:
        """Initialize FullCatalog.

        Args:
            height: In pixels, without accounting for border padding.
            width: In pixels, without accounting for border padding.
            d: Dictionary containing parameters of FullCatalog with correct shape.
        """
        self.height = height
        self.width = width
        self.plocs = d.pop("plocs")  # pixel distance from top-left corner of inner image.
        self.n_sources = d.pop("n_sources")
        self.batch_size = self.plocs.shape[0]
        self.max_n_sources = self.plocs.shape[1]
        assert self.plocs.ndim == 3
        assert self.plocs.shape[-1] == 2
        assert self.n_sources.max().int() <= self.max_n_sources
        assert self.n_sources.shape == (self.batch_size,)
        super().__init__(**d)

    def __setitem__(self, key: str, item: Tensor) -> None:
        if key not in self.allowed_params:
            raise ValueError(
                f"The key '{key}' is not in the allowed parameters for FullCatalog"
                " (check spelling?)"
            )
        self._validate(item)
        super().__setitem__(key, item)

    def __getitem__(self, key: str) -> Tensor:
        assert isinstance(key, str)
        return super().__getitem__(key)

    def to_dict(self) -> dict[str, Tensor]:
        out = {}
        out["plocs"] = self.plocs
        out["n_sources"] = self.n_sources
        for k, v in self.items():
            out[k] = v
        return out

    def to_tensor_dict(self) -> TensorDict:
        return TensorDict(self.to_dict(), batch_size=[self.batch_size])

    def apply_param_bin(self, pname: str, p_min: float, p_max: float):
        """Apply magnitude bin to given parameters."""
        assert pname in self, f"Parameter '{pname}' required to apply mag cut."
        assert self[pname].shape[-1] == 1, "Can only be applied to scalar parameters."
        assert self[pname].min() >= 0, f"Cannot use this function with {pname}."
        assert p_min >= 0, "`p_min` should be at least 0 "

        # get indices to collect
        keep = torch.logical_and(self[pname] < p_max, self[pname] > p_min)
        n_batches, max_sources, _ = keep.shape
        as_indices = torch.arange(0, max_sources).expand(n_batches, max_sources).unsqueeze(-1)
        to_collect = torch.where(keep, as_indices, torch.ones_like(keep) * max_sources)
        to_collect = to_collect.sort(dim=1)[0]

        # get dictionary with all params (including plocs)
        d = dict(self.items())
        d_new = {}
        d["plocs"] = self.plocs
        for k, v in d.items():
            pdim = v.shape[-1]
            to_collect_v = to_collect.expand(n_batches, max_sources, pdim)
            v_expand = torch.hstack([v, torch.zeros(v.shape[0], 1, v.shape[-1])])
            d_new[k] = torch.gather(v_expand, 1, to_collect_v)
        d_new["n_sources"] = keep.sum(dim=(-2, -1)).long()
        return type(self)(self.height, self.width, d_new)

    @property
    def device(self):
        return self.plocs.device

    def to_tile_params(  # noqa: WPS231
        self, tile_slen: int, ignore_extra_sources=False
    ) -> TileCatalog:
        """Returns the TileCatalog (with at most 1 source per tile) for this FullCatalog.

        Args:
            tile_slen: The side length of the tiles.
            ignore_extra_sources: If False (default), raises an error if a tile contains more
                than once source. If True, only adds the tile parameters of the brightest source.

        Returns:
            TileCatalog corresponding to the each source in the FullCatalog.

        Raises:
            ValueError: If the number of sources in one tile is larger than 1 and
                `ignore_extra_sources` is False.
        """
        tile_coords = torch.div(self.plocs, tile_slen, rounding_mode="trunc").to(torch.int)
        nth, ntw = get_n_tiles_hw(self.height, self.width, tile_slen)

        # prepare tiled tensors
        tile_cat_shape = (self.batch_size, nth, ntw)
        tile_locs = torch.zeros((*tile_cat_shape, 2), device=self.device)
        tile_n_sources = torch.zeros(tile_cat_shape, dtype=torch.int64, device=self.device)
        if ignore_extra_sources:
            # need to compare fluxes per tile to get brightest object
            tile_fluxes = torch.zeros(tile_cat_shape, device=self.device)
            assert "fluxes" in self.keys(), "Fluxes are required to decide which source to keep!"

        tile_params: Dict[str, Tensor] = {}
        for k, v in self.items():
            dtype = torch.int64 if k == "objid" else torch.float
            size = (self.batch_size, nth, ntw, v.shape[-1])
            tile_params[k] = torch.zeros(size, dtype=dtype, device=self.device)

        # fill up the tiled tensors
        for ii in range(self.batch_size):
            n_sources = self.n_sources[ii].int()
            assert n_sources.ndim == 0
            for idx, coords in enumerate(tile_coords[ii][:n_sources]):
                n_sources_in_tile = tile_n_sources[ii, coords[0], coords[1]]
                assert n_sources_in_tile.ndim == 0
                assert n_sources_in_tile.le(1) or n_sources_in_tile.ge(0)
                assert n_sources_in_tile.dtype is torch.int64
                if n_sources_in_tile > 0:
                    if not ignore_extra_sources:
                        raise ValueError(  # noqa: WPS220
                            "# of sources in at least one tile is larger than 1."
                        )
                    flux1 = rearrange(tile_fluxes[ii, coords[0], coords[1]], "->")
                    flux2 = rearrange(self["fluxes"][ii, idx], "1 ->")
                    if flux1 > flux2:  # keep current source in tile
                        continue  # noqa: WPS220
                tile_loc = (self.plocs[ii, idx] - coords * tile_slen) / tile_slen
                tile_locs[ii, coords[0], coords[1]] = tile_loc
                for p, q in tile_params.items():
                    q[ii, coords[0], coords[1], :] = self[p][ii, idx]
                tile_n_sources[ii, coords[0], coords[1]] = 1
                if ignore_extra_sources:
                    flux = rearrange(self["fluxes"][ii, idx], "1->")
                    tile_fluxes[ii, coords[0], coords[1]] = flux
        tile_params.update({"locs": tile_locs, "n_sources": tile_n_sources})
        return TileCatalog(tile_slen, tile_params)

    def _validate(self, x: Tensor):
        assert isinstance(x, Tensor)
        assert x.ndim == 3
        assert x.shape[:-1] == (self.batch_size, self.max_n_sources)
        assert x.device == self.device


def stack_full_catalogs(full_cats: List[FullCatalog]) -> FullCatalog:
    all_tds = []
    for full_cat in full_cats:
        all_tds.append(full_cat.to_tensor_dict())
    return torch.cat(all_tds, 0)


def index_full_catalog(
    full_cat: FullCatalog, idx1: int | None = None, idx2: int | None = None
) -> FullCatalog:
    new_td = full_cat.to_tensor_dict()[idx1:idx2]
    return FullCatalog(full_cat.height, full_cat.width, {**new_td})


def get_n_tiles_hw(height: int, width: int, tile_slen: int) -> tuple[int, int]:
    return math.ceil(height / tile_slen), math.ceil(width / tile_slen)


def get_is_on_from_n_sources(n_sources: Tensor, max_n_sources: int) -> Tensor:
    """Provides a tensor indicating which sources are "on" or "off".

    Return a boolean array of `shape=(*n_sources.shape, max_n_sources)` whose `(*,l)th` entry
    indicates whether there are more than l sources on the `*th` index.

    Arguments:
        n_sources: Tensor with number of sources per tile.
        max_n_sources: Maximum number of sources allowed per tile.

    Returns:
        Tensor indicating how many sources are present for each batch.
    """
    assert not torch.any(torch.isnan(n_sources))
    assert torch.all(n_sources >= 0) and torch.all(n_sources <= 1)
    assert torch.all(n_sources.le(max_n_sources))

    is_on_array = torch.zeros(
        *n_sources.shape,
        max_n_sources,
        device=n_sources.device,
        dtype=torch.float,
    )

    for i in range(max_n_sources):
        is_on_array[..., i] = n_sources > i

    return is_on_array
