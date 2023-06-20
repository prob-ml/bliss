import math
from collections import UserDict
from copy import copy
from typing import Dict, Tuple

import torch
from einops import rearrange, reduce, repeat
from matplotlib.pyplot import Axes
from torch import Tensor


class TileCatalog(UserDict):
    allowed_params = {
        "n_source_log_probs",
        "fluxes",
        "star_fluxes",
        "star_log_fluxes",
        "mags",
        "ellips",
        "snr",
        "blendedness",
        "galaxy_bools",
        "galaxy_params",
        "galaxy_fluxes",
        "galaxy_probs",
        "galaxy_blends",
        "star_bools",
        "objid",
        "hlr",
        "ra",
        "dec",
        "matched",
        "mismatched",
        "detection_thresholds",
        "lensed_galaxy_bools",
        "lensed_galaxy_probs",
        "lens_params",
        "log_flux_sd",
        "loc_sd",
    }

    def __init__(self, tile_slen: int, d: Dict[str, Tensor]):
        self.tile_slen = tile_slen
        d = copy(d)  # shallow copy, so we don't mutate the argument
        self.locs = d.pop("locs")
        self.n_sources = d.pop("n_sources")
        self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources = self.locs.shape[:-1]
        super().__init__(**d)

    def __setitem__(self, key: str, item: Tensor) -> None:
        if key not in self.allowed_params:
            msg = f"The key '{key}' is not in the allowed parameters for TileCatalog"
            raise ValueError(msg)
        self._validate(item)
        super().__setitem__(key, item)

    def __getitem__(self, key: str) -> Tensor:
        assert isinstance(key, str)
        return super().__getitem__(key)

    def _validate(self, x: Tensor):
        assert isinstance(x, Tensor)
        assert x.shape[:4] == (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources)
        assert x.device == self.device

    @property
    def is_on_array(self) -> Tensor:
        """Returns a (n x nth x ntw x n_sources) tensor indicating whether source is on."""
        return get_is_on_from_n_sources(self.n_sources, self.max_sources)

    def to(self, device):
        out = {}
        for k, v in self.to_dict().items():
            out[k] = v.to(device)
        return type(self)(self.tile_slen, out)

    @property
    def device(self):
        return self.locs.device

    def crop(self, hlims_tile, wlims_tile):
        out = {}
        for k, v in self.to_dict().items():
            out[k] = v[:, hlims_tile[0] : hlims_tile[1], wlims_tile[0] : wlims_tile[1]]
        return type(self)(self.tile_slen, out)

    def symmetric_crop(self, tiles_to_crop):
        _batch_size, tile_height, tile_width = self.n_sources.shape
        return self.crop(
            [tiles_to_crop, tile_height - tiles_to_crop],
            [tiles_to_crop, tile_width - tiles_to_crop],
        )

    def to_full_params(self):
        """Converts image parameters in tiles to parameters of full image.

        By parameters, we mean samples from the variational distribution, not the variational
        parameters.

        Returns:
            The FullCatalog instance corresponding to the TileCatalog instance.

            NOTE: The locations (`"locs"`) are between 0 and 1. The output also contains
            pixel locations ("plocs") that are between 0 and `slen`.
        """
        plocs = self.get_full_locs_from_tiles()
        param_names_to_mask = {"plocs"}.union(set(self.keys()))
        tile_params_to_gather = {"plocs": plocs}
        tile_params_to_gather.update(self)

        params = {}
        indices_to_retrieve, is_on_array = self.get_indices_of_on_sources()
        for param_name, tile_param in tile_params_to_gather.items():
            k = tile_param.shape[-1]
            param = rearrange(tile_param, "b nth ntw s k -> b (nth ntw s) k", k=k)
            indices_for_param = repeat(indices_to_retrieve, "b nth_ntw_s -> b nth_ntw_s k", k=k)
            param = torch.gather(param, dim=1, index=indices_for_param)
            if param_name in param_names_to_mask:
                param = param * is_on_array.unsqueeze(-1)
            params[param_name] = param

        params["n_sources"] = reduce(self.n_sources, "b nth ntw -> b", "sum")
        height, width = self.n_tiles_h * self.tile_slen, self.n_tiles_w * self.tile_slen
        return FullCatalog(height, width, params)

    def get_full_locs_from_tiles(self) -> Tensor:
        """Get the full image locations from tile locations.

        Returns:
            Tensor: pixel coordinates of each source (between 0 and slen).
        """
        slen = self.n_tiles_h * self.tile_slen
        wlen = self.n_tiles_w * self.tile_slen
        # coordinates on tiles.
        x_coords = torch.arange(0, slen, self.tile_slen, device=self.locs.device).long()
        y_coords = torch.arange(0, wlen, self.tile_slen, device=self.locs.device).long()
        tile_coords = torch.cartesian_prod(x_coords, y_coords)

        # recenter and renormalize locations.
        locs = rearrange(self.locs, "b nth ntw d xy -> (b nth ntw) d xy", xy=2)
        bias = repeat(tile_coords, "n xy -> (r n) 1 xy", r=self.batch_size).float()

        plocs = locs * self.tile_slen + bias
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
            This is accomplishied by flattening the n_tiles_per_image and max_detections.
            For example, if we had 3 tiles with a maximum of two sources each,
            the elements of this tensor would take values from 0 up to and including 5.
        """
        tile_is_on_array_sampled = self.is_on_array
        n_sources = reduce(tile_is_on_array_sampled, "b nth ntw d -> b", "sum")
        max_sources = int(n_sources.max().int().item())
        tile_is_on_array = rearrange(tile_is_on_array_sampled, "b nth ntw d -> b (nth ntw d)")
        indices_sorted = tile_is_on_array.long().argsort(dim=1, descending=True)
        indices_sorted = indices_sorted[:, :max_sources]

        is_on_array = torch.gather(tile_is_on_array, dim=1, index=indices_sorted)
        return indices_sorted, is_on_array

    def to_dict(self) -> Dict[str, Tensor]:
        out = {}
        out["locs"] = self.locs
        out["n_sources"] = self.n_sources
        for k, v in self.items():
            out[k] = v
        return out

    def gather_param_at_tiles(self, param_name: str, indices: Tensor) -> Tensor:
        """Gets the tile parameters at the desired indices.

        Args:
            param_name (str): Param name. Must be either "locs" or in keys of catalog.
            indices (Tensor): The indices to gather. Normally this will come from
                get_indices_of_on_sources, but notably, we use the indices of the true catalog to
                gather from the tile catalog when computing metrics.

        Returns:
            Tensor: (b, n, k) tensor, where b=batch size, n=length of indices, and k=param dims
        """
        assert param_name == "locs" or param_name in self.keys(), f"'{param_name}' not in catalog"
        param = self.get_full_locs_from_tiles() if param_name == "locs" else self[param_name]
        param = rearrange(param, "b h w s k -> b (h w s) k")
        idx_to_gather = repeat(indices, "... -> ... k", k=param.size(-1))
        return torch.gather(param, dim=1, index=idx_to_gather)

    def _get_fluxes_of_on_sources(self):
        """Gets fluxes of "on" sources based on whether the source is a star or galaxy.

        Returns:
            Tensor: a tensor of fluxes of size (b x nth x ntw x max_sources x 1)
        """
        fluxes = torch.where(
            self["galaxy_bools"], self["galaxy_params"][..., 0, None], self["star_fluxes"]
        )
        return torch.where(self.is_on_array[..., None], fluxes, torch.zeros_like(fluxes))

    def get_brightest_source_per_tile(self):
        """Restrict TileCatalog to only the brightest 'on' source per tile.

        Returns:
            TileCatalog: a new catalog with only one source per tile
        """
        if self.max_sources == 1:
            return self

        # sort by fluxes of "on" sources to get brightest source per tile
        on_fluxes = self._get_fluxes_of_on_sources()
        top_idx = on_fluxes.argsort(dim=3, descending=True)[
            :, :, :, 0:1, 0
        ]  # 0:1 keeps dims right for slicing

        d = {}
        for key, val in self.to_dict().items():
            if key == "n_sources":
                d[key] = self.n_sources.bool().int()  # send all positive values to 1, 0 to 0
            else:
                param_dim = val.size(-1)
                idx_to_gather = repeat(top_idx, "... -> ... k", k=param_dim)
                d[key] = torch.take_along_dim(val, idx_to_gather, dim=3)

        return TileCatalog(self.tile_slen, d)

    def filter_tile_catalog_by_flux(self, min_flux=0, max_flux=torch.inf):
        """Restricts TileCatalog to sources that have a flux between min_flux and max_flux.

        Args:
            min_flux (float): Minimum flux value to keep. Defaults to 622.
            max_flux (float): Maximum flux value to keep. Defaults to 1e6.

        Returns:
            TileCatalog: a new catalog with only sources within the flux range. Note that the size
                of the tensors stays the same, but params at sources outside the range are set to 0.
        """

        # get fluxes of "on" sources to mask by
        on_fluxes = self._get_fluxes_of_on_sources()
        flux_mask = (on_fluxes > min_flux) & (on_fluxes < max_flux)

        d = {}
        for key, val in self.to_dict().items():
            if key == "n_sources":
                d[key] = flux_mask.sum(dim=3).squeeze()  # number of sources within range in tile
            else:
                d[key] = torch.where(flux_mask, val, torch.zeros_like(val))

        return TileCatalog(self.tile_slen, d)


class FullCatalog(UserDict):
    allowed_params = TileCatalog.allowed_params

    def __init__(self, height: int, width: int, d: Dict[str, Tensor]) -> None:
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
        self.batch_size, self.max_sources, hw = self.plocs.shape
        assert hw == 2
        assert self.n_sources.max().int().item() <= self.max_sources
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

    def _validate(self, x: Tensor):
        assert isinstance(x, Tensor)
        assert x.shape[:-1] == (self.batch_size, self.max_sources)
        assert x.device == self.device

    def to_dict(self) -> Dict[str, Tensor]:
        out = {}
        out["plocs"] = self.plocs
        out["n_sources"] = self.n_sources
        for k, v in self.items():
            out[k] = v
        return out

    def to(self, device):
        out = {}
        for k, v in self.to_dict().items():
            out[k] = v.to(device)
        return type(self)(self.height, self.width, out)

    @property
    def device(self):
        return self.plocs.device

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
        d["plocs"] = self.plocs
        for k, v in d.items():
            pdim = v.shape[-1]
            to_collect_v = to_collect.expand(n_batches, max_sources, pdim)
            v_expand = torch.hstack([v, torch.zeros(v.shape[0], 1, v.shape[-1])])
            d_new[k] = torch.gather(v_expand, 1, to_collect_v)
        d_new["n_sources"] = keep.sum(dim=(-2, -1)).long()
        return type(self)(self.height, self.width, d_new)

    def to_tile_params(
        self, tile_slen: int, max_sources_per_tile: int, ignore_extra_sources=False
    ) -> TileCatalog:
        """Returns the TileCatalog corresponding to this FullCatalog.

        Args:
            tile_slen: The side length of the tiles.
            max_sources_per_tile: The maximum number of sources in one tile.
            ignore_extra_sources: If False (default), raises an error if the number of sources
                in one tile exceeds the `max_sources_per_tile`. If True, only adds the tile
                parameters of the first `max_sources_per_tile` sources to the new TileCatalog.

        Returns:
            TileCatalog correspond to the each source in the FullCatalog.

        Raises:
            ValueError: If the number of sources in one tile exceeds `max_sources_per_tile` and
                `ignore_extra_sources` is False.
        """
        tile_coords = torch.div(self.plocs, tile_slen, rounding_mode="trunc").to(torch.int)
        n_tiles_h, n_tiles_w = get_n_tiles_hw(self.height, self.width, tile_slen)

        # prepare tiled tensors
        tile_cat_shape = (self.batch_size, n_tiles_h, n_tiles_w, max_sources_per_tile)
        tile_locs = torch.zeros((*tile_cat_shape, 2), device=self.device)
        tile_n_sources = torch.zeros(tile_cat_shape[:3], dtype=torch.int64, device=self.device)
        tile_params: Dict[str, Tensor] = {}
        for k, v in self.items():
            dtype = torch.int64 if k == "objid" else torch.float
            size = (self.batch_size, n_tiles_h, n_tiles_w, max_sources_per_tile, v.shape[-1])
            tile_params[k] = torch.zeros(size, dtype=dtype, device=self.device)

        for ii in range(self.batch_size):
            n_sources = int(self.n_sources[ii].item())
            for idx, coords in enumerate(tile_coords[ii][:n_sources]):
                source_idx = tile_n_sources[ii, coords[0], coords[1]].item()
                if source_idx >= max_sources_per_tile:
                    if not ignore_extra_sources:
                        raise ValueError(  # noqa: WPS220
                            "# of sources per tile exceeds `max_sources_per_tile`."
                        )
                    continue  # ignore extra sources in this tile.
                tile_loc = (self.plocs[ii, idx] - coords * tile_slen) / tile_slen
                tile_locs[ii, coords[0], coords[1], source_idx] = tile_loc
                for k, v in tile_params.items():
                    v[ii, coords[0], coords[1], source_idx] = self[k][ii, idx]
                tile_n_sources[ii, coords[0], coords[1]] = source_idx + 1
        tile_params.update({"locs": tile_locs, "n_sources": tile_n_sources})
        return TileCatalog(tile_slen, tile_params)

    def plot_plocs(self, ax: Axes, idx: int, object_type: str, bp: int = 0, **kwargs):
        if object_type == "galaxy":
            keep = self["galaxy_bools"][idx, :].squeeze(-1).bool()
        elif object_type == "star":
            keep = self["star_bools"][idx, :].squeeze(-1).bool()
        elif object_type == "all":
            keep = torch.ones(self.max_sources, dtype=torch.bool, device=self.plocs.device)
        else:
            raise NotImplementedError()
        plocs = self.plocs[idx, keep] - 0.5 + bp
        plocs = plocs.detach().cpu()
        ax.scatter(plocs[:, 1], plocs[:, 0], **kwargs)


def get_n_tiles_hw(height: int, width: int, tile_slen: int):
    return math.ceil(height / tile_slen), math.ceil(width / tile_slen)


def get_is_on_from_n_sources(n_sources, max_sources):
    """Provides tensor which indicates how many sources are present for each batch.

    Return a boolean array of `shape=(*n_sources.shape, max_sources)` whose `(*,l)th` entry
    indicates whether there are more than l sources on the `*th` index.

    Arguments:
        n_sources: Tensor with number of sources per tile.
        max_sources: Maximum number of sources allowed per tile.

    Returns:
        Tensor indicating how many sources are present for each batch.
    """
    assert not torch.any(torch.isnan(n_sources))
    assert torch.all(n_sources >= 0)
    assert torch.all(n_sources.le(max_sources))

    is_on_array = torch.zeros(
        *n_sources.shape,
        max_sources,
        device=n_sources.device,
        dtype=torch.bool,
    )

    for i in range(max_sources):
        is_on_array[..., i] = n_sources > i

    return is_on_array
