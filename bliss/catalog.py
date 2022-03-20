from collections import UserDict
from typing import Dict, Tuple

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor
from torch.nn import functional as F


class TileCatalog(UserDict):
    allowed_params = {
        "fluxes",
        "log_fluxes",
        "mags",
        "galaxy_bools",
        "galaxy_params",
        "galaxy_fluxes",
        "galaxy_probs",
        "galaxy_blends",
        "star_bools",
    }

    def __init__(self, tile_slen: int, d: Dict[str, Tensor]):
        self.tile_slen = tile_slen
        self.locs = d.pop("locs")
        self.n_sources = d.pop("n_sources")
        self.n_sources_log_prob = d.pop("n_sources_log_prob", None)

        self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources = self.locs.shape[:-1]
        assert self.n_sources.shape == (self.batch_size, self.n_tiles_h, self.n_tiles_w)
        super().__init__(**d)

    def __setitem__(self, key: str, item: Tensor) -> Tensor:
        if key not in self.allowed_params:
            raise ValueError(
                f"The key '{key}' is not in the allowed parameters for TileCatalog"
                " (check spelling?)"
            )
        self._validate(item)
        super().__setitem__(key, item)

    def _validate(self, x: Tensor):
        assert isinstance(x, Tensor)
        assert x.shape[:4] == (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources)

    @property
    def is_on_array(self) -> Tensor:
        """Returns a n x nth x ntw x n_sources tensor indicating whether source is on."""
        return get_is_on_from_n_sources(self.n_sources, self.max_sources)

    def cpu(self):
        return self.to("cpu")

    def to(self, device):
        out = {}
        for k, v in self.to_dict().items():
            out[k] = v.to(device)
        return type(self)(self.tile_slen, out)

    def get_full_params(self) -> Dict[str, Tensor]:
        """Converts image parameters in tiles to parameters of full image.

        By parameters, we mean samples from the variational distribution, not the variational
        parameters.

        Returns:
            A dictionary of tensors with the same members as those in `tile_params`.
            The first two dimensions of each tensor is `batch_size x max(n_sources)`,
            where `max(n_sources)` is the maximum number of sources detected across samples.
            In samples where the number of sources detected is less than max(n_sources),
            these values will be zeroed out. Thus, it is imperative to use the "n_sources"
            element to verify which locations/fluxes/parameters are zeroed out.

            NOTE: The locations (`"locs"`) are between 0 and 1. The output also contains
            pixel locations ("plocs") that are between 0 and slen.
        """
        plocs, locs = self._get_full_locs_from_tiles()
        param_names_to_mask = {"locs", "plocs"}.union(set(self.keys()))
        tile_params_to_gather = {
            "locs": locs,
            "plocs": plocs,
        }
        tile_params_to_gather.update(self)

        params = {}
        indices_to_retrieve, is_on_array = self._get_indices_of_on_sources()
        for param_name, tile_param in tile_params_to_gather.items():
            k = tile_param.shape[-1]
            param = rearrange(tile_param, "b nth ntw s k -> b (nth ntw s) k", k=k)
            indices_for_param = repeat(indices_to_retrieve, "b nth_ntw_s -> b nth_ntw_s k", k=k)
            param = torch.gather(param, dim=1, index=indices_for_param)
            if param_name in param_names_to_mask:
                param *= is_on_array.unsqueeze(-1)
            params[param_name] = param

        params["n_sources"] = reduce(self.n_sources, "b nth ntw -> b", "sum")
        assert params["locs"].shape[1] == params["n_sources"].max().int().item()
        return params

    def _get_full_locs_from_tiles(self) -> Tuple[Tensor, Tensor]:
        """Get the full image locations from tile locations.

        Returns:
            A tuple of two elements, "plocs" and "locs".
            "plocs" are the pixel coordinates of each source (between 0 and slen).
            "locs" are the scaled locations of each source (between 0 and 1).
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
        plocs = rearrange(
            plocs,
            "(b nth ntw) d xy -> b nth ntw d xy",
            b=self.batch_size,
            nth=self.n_tiles_h,
            ntw=self.n_tiles_w,
        )
        locs = plocs.clone()
        locs[..., 0] /= slen
        locs[..., 1] /= wlen

        return plocs, locs

    def _get_indices_of_on_sources(self) -> Tensor:
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
        tile_is_on_array_sampled = get_is_on_from_n_sources(self.n_sources, self.max_sources)
        n_sources = reduce(tile_is_on_array_sampled, "b nth ntw d -> b", "sum")
        max_sources = n_sources.max().int().item()
        tile_is_on_array = rearrange(tile_is_on_array_sampled, "b nth ntw d -> b (nth ntw d)")
        indices_sorted = tile_is_on_array.long().argsort(dim=1, descending=True)
        indices_sorted = indices_sorted[:, :max_sources]

        is_on_array = torch.gather(tile_is_on_array, dim=1, index=indices_sorted)
        return indices_sorted, is_on_array

    def to_dict(self) -> Dict[str, Tensor]:
        out = {}
        out["locs"] = self.locs
        out["n_sources"] = self.n_sources
        if self.n_sources_log_prob is not None:
            out["n_sources_log_prob"] = self.n_sources_log_prob
        for k, v in self.items():
            out[k] = v
        return out

    def equals(self, other, exclude=None):
        self_dict = self.to_dict()
        other_dict: Dict[str, Tensor] = other.to_dict()
        exclude = set() if exclude is None else set(exclude)
        keys = set(self_dict.keys()).union(other_dict.keys()).difference(exclude)
        for k in keys:
            if not torch.allclose(self_dict[k], other_dict[k]):
                return False
        return True

    def __eq__(self, other):
        return self.equals(other)


def get_tile_params_from_full(
    full_params: Dict[str, Tensor], tile_slen: int, n_tiles_h, n_tiles_w, max_sources
):
    full_plocs = full_params["plocs"]
    batch_size, n_sources, v = full_plocs.shape
    assert batch_size == 1
    tile_coords = (full_plocs // tile_slen).to(torch.int).squeeze(0)

    tile_locs = torch.zeros((batch_size, n_tiles_h, n_tiles_w, max_sources, 2))
    tile_n_sources = torch.zeros((batch_size, n_tiles_h, n_tiles_w), dtype=torch.int64)
    tile_is_on_array = torch.zeros((batch_size, n_tiles_h, n_tiles_w, max_sources, 1))
    param_names_to_gather = {
        "galaxy_bools",
        "star_bools",
        "galaxy_params",
        "fluxes",
        "log_fluxes",
        "galaxy_fluxes",
        "galaxy_probs",
        "galaxy_blends",
    }
    tile_params = {}
    for k, v in full_params.items():
        if k in param_names_to_gather:
            dim = v.shape[-1]
            tile_params[k] = torch.zeros((batch_size, n_tiles_h, n_tiles_w, max_sources, dim))
    n_sources = full_params["n_sources"][0]
    for (idx, coords) in enumerate(tile_coords[:n_sources]):
        source_idx = tile_n_sources[0, coords[0], coords[1]]
        tile_n_sources[0, coords[0], coords[1]] = source_idx + 1
        tile_is_on_array[0, coords[0], coords[1]] = 1
        tile_locs[0, coords[0], coords[1], source_idx] = full_plocs[0, idx] - coords * tile_slen
        for k, v in tile_params.items():
            v[0, coords[0], coords[1], source_idx] = full_params[k][0, idx]
    tile_params.update(
        {
            "locs": tile_locs,
            "n_sources": tile_n_sources,
        }
    )
    return TileCatalog(tile_slen, tile_params)


def get_images_in_tiles(images, tile_slen, ptile_slen):
    """Divides a batch of full images into padded tiles.

    This is similar to nn.conv2d, with a sliding window=ptile_slen and stride=tile_slen.

    Arguments:
        images: Tensor of images with size (batchsize x n_bands x slen x slen)
        tile_slen: Side length of tile
        ptile_slen: Side length of padded tile

    Returns:
        A batchsize x n_tiles_h x n_tiles_w x n_bands x tile_weight x tile_width image
    """
    assert len(images.shape) == 4
    n_bands = images.shape[1]
    window = ptile_slen
    n_tiles_h, n_tiles_w = get_n_tiles_hw(images.shape[2], images.shape[3], window, tile_slen)
    tiles = F.unfold(images, kernel_size=window, stride=tile_slen)
    # b: batch, c: channel, h: tile height, w: tile width, n: num of total tiles for each batch
    return rearrange(
        tiles,
        "b (c h w) (nth ntw) -> b nth ntw c h w",
        nth=n_tiles_h,
        ntw=n_tiles_w,
        c=n_bands,
        h=window,
        w=window,
    )


def get_n_tiles_hw(h, w, window, tile_slen):
    nh = ((h - window) // tile_slen) + 1
    nw = ((w - window) // tile_slen) + 1
    return nh, nw


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
        dtype=torch.float,
    )

    for i in range(max_sources):
        is_on_array[..., i] = n_sources > i

    return is_on_array
