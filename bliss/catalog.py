import math
from collections import UserDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from einops import rearrange, reduce, repeat
from matplotlib.pyplot import Axes
from torch import Tensor
from torch.nn import functional as F

from bliss.datasets.sdss import SloanDigitalSkySurvey, column_to_tensor, convert_flux_to_mag


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
    }

    def __init__(self, tile_slen: int, d: Dict[str, Tensor]):
        self.tile_slen = tile_slen
        self.locs = d.pop("locs")
        self.n_sources = d.pop("n_sources")
        self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources = self.locs.shape[:-1]
        assert self.n_sources.shape == (self.batch_size, self.n_tiles_h, self.n_tiles_w)
        super().__init__(**d)

    def __setitem__(self, key: str, item: Tensor) -> None:
        if key not in self.allowed_params:
            raise ValueError(
                f"The key '{key}' is not in the allowed parameters for TileCatalog"
                " (check spelling?)"
            )
        self._validate(item)
        super().__setitem__(key, item)

        # provide star_bools automatically if galaxy_bools is set.
        if key == "galaxy_bools" and "star_bools" not in self:
            star_bools = self.is_on_array.unsqueeze(-1) * (1 - item)
            super().__setitem__("star_bools", star_bools)

    def __getitem__(self, key: str) -> Tensor:
        assert isinstance(key, str)
        return super().__getitem__(key)

    def _validate(self, x: Tensor):
        assert isinstance(x, Tensor)
        assert x.shape[:4] == (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources)
        assert x.device == self.device

    @classmethod
    def from_flat_dict(cls, tile_slen: int, n_tiles_h: int, n_tiles_w: int, d: Dict[str, Tensor]):
        catalog_dict: Dict[str, Tensor] = {}
        for k, v in d.items():
            catalog_dict[k] = v.reshape(-1, n_tiles_h, n_tiles_w, *v.shape[1:])
        return cls(tile_slen, catalog_dict)

    @property
    def is_on_array(self) -> Tensor:
        """Returns a (n x nth x ntw x n_sources) tensor indicating whether source is on."""
        return get_is_on_from_n_sources(self.n_sources, self.max_sources)

    def cpu(self):
        return self.to("cpu")

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

    def to_full_params(self):
        """Converts image parameters in tiles to parameters of full image.

        By parameters, we mean samples from the variational distribution, not the variational
        parameters.

        Returns:
            The FullCatalog instance corresponding to the TileCatalog instance.

            NOTE: The locations (`"locs"`) are between 0 and 1. The output also contains
            pixel locations ("plocs") that are between 0 and `slen`.
        """
        plocs = self._get_full_locs_from_tiles()
        param_names_to_mask = {"plocs"}.union(set(self.keys()))
        tile_params_to_gather = {"plocs": plocs}
        tile_params_to_gather.update(self)

        params = {}
        indices_to_retrieve, is_on_array = self._get_indices_of_on_sources()
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

    def _get_full_locs_from_tiles(self) -> Tensor:
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
        return rearrange(
            plocs,
            "(b nth ntw) d xy -> b nth ntw d xy",
            b=self.batch_size,
            nth=self.n_tiles_h,
            ntw=self.n_tiles_w,
        )

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

    def equals(self, other, exclude=None, **kwargs) -> bool:
        self_dict = self.to_dict()
        other_dict: Dict[str, Tensor] = other.to_dict()
        exclude = set() if exclude is None else set(exclude)
        keys = set(self_dict.keys()).union(other_dict.keys()).difference(exclude)
        for k in keys:
            if not torch.allclose(self_dict[k], other_dict[k], **kwargs):
                return False
        return True

    def __eq__(self, other):
        return self.equals(other)

    def get_tile_params_at_coord(self, plocs: torch.Tensor) -> Dict[str, Tensor]:
        """Return the parameters of the tiles that contain each of the locations in plocs."""
        assert len(plocs.shape) == 2 and plocs.shape[1] == 2
        assert plocs.device == self.locs.device
        n_total = len(plocs)
        slen = self.n_tiles_h * self.tile_slen
        wlen = self.n_tiles_w * self.tile_slen
        # coordinates on tiles.
        x_coords = torch.arange(0, slen, self.tile_slen, device=self.locs.device).long()
        y_coords = torch.arange(0, wlen, self.tile_slen, device=self.locs.device).long()

        x_indx = torch.searchsorted(x_coords.contiguous(), plocs[:, 0].contiguous()) - 1
        y_indx = torch.searchsorted(y_coords.contiguous(), plocs[:, 1].contiguous()) - 1

        return {k: v[:, x_indx, y_indx, :, :].reshape(n_total, -1) for k, v in self.items()}

    def set_all_fluxes_and_mags(self, decoder) -> None:
        """Set all fluxes (galaxy and star) of tile catalog given an `ImageDecoder` instance."""
        # first get galaxy fluxes
        assert "galaxy_bools" in self and "galaxy_params" in self and "star_fluxes" in self
        assert (
            self.device == decoder.device
        ), f"TileCatalog on {self.device} but decoder on {decoder.device}"
        galaxy_bools, galaxy_params = self["galaxy_bools"], self["galaxy_params"]
        with torch.no_grad():
            galaxy_fluxes = decoder.get_galaxy_fluxes(galaxy_bools, galaxy_params)
        self["galaxy_fluxes"] = galaxy_fluxes

        # update default fluxes
        star_bools = self["star_bools"]
        star_fluxes = self["star_fluxes"]
        fluxes = galaxy_bools * galaxy_fluxes + star_bools * star_fluxes
        self["fluxes"] = fluxes

        # mags (careful with 0s)
        is_on_array = self.is_on_array > 0
        self["mags"] = torch.zeros_like(self["fluxes"])
        self["mags"][is_on_array] = convert_flux_to_mag(self["fluxes"][is_on_array])

    def set_galaxy_ellips(self, decoder, scale: float = 0.393) -> None:
        """Sets galaxy ellipticities of tile catalog given an `ImageDecoder` instance."""
        galaxy_bools, galaxy_params = self["galaxy_bools"], self["galaxy_params"]
        ellips = decoder.get_galaxy_ellips(galaxy_bools, galaxy_params, scale=scale)
        self["ellips"] = ellips


class FullCatalog(UserDict):
    allowed_params = TileCatalog.allowed_params

    def __init__(self, height: int, width: int, d: Dict[str, Tensor]):
        self.height = height
        self.width = width
        self.plocs = d.pop("plocs")
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

        # provide star_bools automatically if galaxy_bools is set.
        if key == "galaxy_bools" and "star_bools" not in self:
            assert item.dtype != torch.bool, "galaxy_bools should be float for consistency."
            is_on_array = get_is_on_from_n_sources(self.n_sources, self.max_sources)
            star_bools = (1 - item) * is_on_array.unsqueeze(-1)
            super().__setitem__("star_bools", star_bools)

    def __getitem__(self, key: str) -> Tensor:
        assert isinstance(key, str)
        return super().__getitem__(key)

    def _validate(self, x: Tensor):
        assert isinstance(x, Tensor)
        assert x.shape[:-1] == (self.batch_size, self.max_sources)
        assert x.device == self.device

    @property
    def device(self):
        return self.plocs.device

    def equals(self, other, exclude=None):
        assert self.batch_size == other.batch_size == 1
        idx_self = self.plocs[0, :, 0].argsort()
        idx_other: Tensor = other.plocs[0, :, 0].argsort()
        exclude = set() if exclude is None else set(exclude)
        keys = set(self.keys()).union(other.keys()).difference(exclude)
        if not torch.allclose(self.plocs[:, idx_self, :], other.plocs[:, idx_other, :]):
            return False
        for k in keys:
            self_value = self[k][:, idx_self, :].float()
            other_value = other[k][:, idx_other, :].float()
            if not torch.allclose(self_value, other_value, equal_nan=True):
                return False
        return True

    def crop(
        self,
        h_min: Optional[int] = None,
        h_max: Optional[int] = None,
        w_min: Optional[int] = None,
        w_max: Optional[int] = None,
    ):
        assert self.batch_size == 1
        keep = torch.ones(self.max_sources, dtype=torch.bool)
        height_out = self.height
        width_out = self.width
        if h_min is not None:
            keep *= self.plocs[0, :, 0] >= h_min
            height_out -= h_min
        if h_max is not None:
            keep *= self.plocs[0, :, 0] <= (self.height - h_max)
            height_out -= h_max
        if w_min is not None:
            keep *= self.plocs[0, :, 1] >= w_min
            width_out -= w_min
        if w_max is not None:
            keep *= self.plocs[0, :, 1] <= (self.width - w_max)
            width_out -= w_max
        d = {}
        d["plocs"] = self.plocs[:, keep] - torch.tensor(
            [h_min, w_min], dtype=self.plocs.dtype, device=self.plocs.device
        )
        d["n_sources"] = keep.sum().reshape(1).to(self.n_sources.dtype).to(self.n_sources.device)
        for k, v in self.items():
            d[k] = v[:, keep]
        return type(self)(height_out, width_out, d)

    def crop_at_coords(self, h_start, h_end, w_start, w_end):
        h_max = self.height - h_end
        w_max = self.width - w_end
        return self.crop(h_start, h_max, w_start, w_max)

    def apply_param_bin(self, pname: str, p_min: float, p_max: float):
        """Apply magnitude bin to given parameters."""
        assert pname in self, f"Parameter '{pname}' required to apply mag cut."
        assert self[pname].shape[-1] == 1, "Can only be applied to scalar parameters."
        assert self[pname].min().item() >= 0, f"Cannot use this function with {pname}."
        assert p_min >= 0, "`p_min` should bet at least 0 "

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
            for (idx, coords) in enumerate(tile_coords[ii][:n_sources]):
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


class PhotoFullCatalog(FullCatalog):
    """Class for the SDSS PHOTO Catalog.

    Some resources:
    - https://www.sdss.org/dr12/algorithms/classify/
    - https://www.sdss.org/dr12/algorithms/resolve/
    """

    @classmethod
    def from_file(cls, sdss_path, run, camcol, field, band):
        sdss_path = Path(sdss_path)
        camcol_dir = sdss_path / str(run) / str(camcol) / str(field)
        po_path = camcol_dir / f"photoObj-{run:06d}-{camcol:d}-{field:04d}.fits"
        po_fits = fits.getdata(po_path)
        objc_type = column_to_tensor(po_fits, "objc_type")
        thing_id = column_to_tensor(po_fits, "thing_id")
        ras = column_to_tensor(po_fits, "ra")
        decs = column_to_tensor(po_fits, "dec")
        galaxy_bools = (objc_type == 3) & (thing_id != -1)
        star_bools = (objc_type == 6) & (thing_id != -1)
        star_fluxes = column_to_tensor(po_fits, "psfflux") * star_bools.reshape(-1, 1)
        star_mags = column_to_tensor(po_fits, "psfmag") * star_bools.reshape(-1, 1)
        galaxy_fluxes = column_to_tensor(po_fits, "cmodelflux") * galaxy_bools.reshape(-1, 1)
        galaxy_mags = column_to_tensor(po_fits, "cmodelmag") * galaxy_bools.reshape(-1, 1)
        fluxes = star_fluxes + galaxy_fluxes
        mags = star_mags + galaxy_mags
        keep = galaxy_bools | star_bools
        galaxy_bools = galaxy_bools[keep]
        star_bools = star_bools[keep]
        ras = ras[keep]
        decs = decs[keep]
        fluxes = fluxes[keep][:, band]
        mags = mags[keep][:, band]

        sdss = SloanDigitalSkySurvey(sdss_path, run, camcol, fields=(field,), bands=(band,))
        wcs: WCS = sdss[0]["wcs"][0]

        # get pixel coordinates
        pts = []
        prs = []
        for ra, dec in zip(ras, decs):
            pt, pr = wcs.wcs_world2pix(ra, dec, 0)
            pts.append(float(pt))
            prs.append(float(pr))
        pts = torch.tensor(pts) + 0.5  # For consistency with BLISS
        prs = torch.tensor(prs) + 0.5
        plocs = torch.stack((prs, pts), dim=-1)
        nobj = plocs.shape[0]

        d = {
            "plocs": plocs.reshape(1, nobj, 2),
            "n_sources": torch.tensor((nobj,)),
            "galaxy_bools": galaxy_bools.reshape(1, nobj, 1).float(),
            "star_bools": star_bools.reshape(1, nobj, 1).float(),
            "fluxes": fluxes.reshape(1, nobj, 1),
            "mags": mags.reshape(1, nobj, 1),
            "ra": ras.reshape(1, nobj, 1),
            "dec": decs.reshape(1, nobj, 1),
        }

        height = sdss[0]["image"].shape[1]
        width = sdss[0]["image"].shape[2]

        return cls(height, width, d)


class DecalsFullCatalog(FullCatalog):
    """Class for the Decals Sweep Tractor Catalog.

    Some resources:
    - https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/
    - https://www.legacysurvey.org/dr9/files/#sweep-catalogs-region-sweep
    - https://www.legacysurvey.org/dr5/description/#photometry
    - https://www.legacysurvey.org/dr9/bitmasks/
    """

    @staticmethod
    def _flux_to_mag(flux):
        return 22.5 - 2.5 * torch.log10(flux)

    @classmethod
    def from_file(
        cls,
        decals_cat_path,
        wcs: WCS,
        hlim: Tuple[int, int],  # in degrees
        wlim: Tuple[int, int],  # in degrees
        band: str = "r",
        mag_max=23,
    ):
        assert hlim[0] == wlim[0]
        catalog_path = Path(decals_cat_path)
        table = Table.read(catalog_path, format="fits")
        band = band.capitalize()

        ra_lim, dec_lim = wcs.all_pix2world(wlim, hlim, 0)
        bitmask = 0b0011010000000001  # noqa: WPS339

        objid = column_to_tensor(table, "OBJID")
        objc_type = table["TYPE"].data.astype(str)
        bits = table["MASKBITS"].data.astype(int)
        is_galaxy = torch.from_numpy(
            (objc_type == "DEV")
            | (objc_type == "REX")
            | (objc_type == "EXP")
            | (objc_type == "SER")
        )
        is_star = torch.from_numpy(objc_type == "PSF")
        ra = column_to_tensor(table, "RA")
        dec = column_to_tensor(table, "DEC")
        flux = column_to_tensor(table, f"FLUX_{band}")
        mask = torch.from_numpy((bits & bitmask) == 0).bool()

        galaxy_bool = is_galaxy & mask & (flux > 0)
        star_bool = is_star & mask & (flux > 0)

        # get pixel coordinates
        pt, pr = wcs.all_world2pix(ra, dec, 0)  # pixels
        pt = torch.tensor(pt)
        pr = torch.tensor(pr)
        ploc = torch.stack((pr, pt), dim=-1)

        # filter on locations
        # first lims imposed on frame
        keep_coord = (ra > ra_lim[0]) & (ra < ra_lim[1]) & (dec > dec_lim[0]) & (dec < dec_lim[1])
        # then, SDSS saturation
        regions = [(1200, 1360, 1700, 1900), (280, 400, 1220, 1320)]
        keep_sat = torch.ones(len(ra)).bool()
        for lim in regions:
            keep_sat = keep_sat & ((pr < lim[0]) | (pr > lim[1]) | (pt < lim[2]) | (pt > lim[3]))
        keep = (galaxy_bool | star_bool) & keep_coord & keep_sat

        # filter quantities
        objid = objid[keep]
        galaxy_bool = galaxy_bool[keep]
        star_bool = star_bool[keep]
        ra = ra[keep]
        dec = dec[keep]
        flux = flux[keep]
        mag = cls._flux_to_mag(flux)
        ploc = ploc[keep, :]
        nobj = ploc.shape[0]

        d = {
            "objid": objid.reshape(1, nobj, 1),
            "ra": ra.reshape(1, nobj, 1),
            "dec": dec.reshape(1, nobj, 1),
            "plocs": ploc.reshape(1, nobj, 2) - hlim[0] + 0.5,  # BLISS consistency
            "n_sources": torch.tensor((nobj,)),
            "galaxy_bools": galaxy_bool.reshape(1, nobj, 1).float(),
            "star_bools": star_bool.reshape(1, nobj, 1).float(),
            "fluxes": flux.reshape(1, nobj, 1),
            "mags": mag.reshape(1, nobj, 1),
        }

        height = hlim[1] - hlim[0]
        width = wlim[1] - wlim[0]
        full_cat = cls(height, width, d)
        return full_cat.apply_param_bin("mags", 0, mag_max)


def get_images_in_tiles(images: Tensor, tile_slen: int, ptile_slen: int) -> Tensor:
    """Divides a batch of full images into padded tiles.

    This is similar to nn.conv2d, with a sliding window=ptile_slen and stride=tile_slen.

    Arguments:
        images: Tensor of images with size (batchsize x n_bands x slen x slen)
        tile_slen: Side length of tile
        ptile_slen: Side length of padded tile

    Returns:
        A batchsize x n_tiles_h x n_tiles_w x n_bands x tile_height x tile_width image
    """
    assert len(images.shape) == 4
    n_bands = images.shape[1]
    window = ptile_slen
    n_tiles_h, n_tiles_w = get_n_padded_tiles_hw(
        images.shape[2], images.shape[3], window, tile_slen
    )
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


def get_n_tiles_hw(height: int, width: int, tile_slen: int):
    return math.ceil(height / tile_slen), math.ceil(width / tile_slen)


def get_n_padded_tiles_hw(height, width, window, tile_slen):
    nh = ((height - window) // tile_slen) + 1
    nw = ((width - window) // tile_slen) + 1
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
