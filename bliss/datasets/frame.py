import math
from pathlib import Path

import numpy as np
import torch

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.sdss import SloanDigitalSkySurvey, convert_flux_to_mag
from bliss.datasets.simulated import SimulatedDataset
from bliss.reporting import CoaddFullCatalog


class SDSSFrame:
    def __init__(self, sdss_dir: str, pixel_scale: float, coadd_file: str):
        run = 94
        camcol = 1
        field = 12
        bands = (2,)
        sdss_data = SloanDigitalSkySurvey(
            sdss_dir=sdss_dir,
            run=run,
            camcol=camcol,
            fields=(field,),
            bands=bands,
        )
        self.data = sdss_data[0]
        self.wcs = self.data["wcs"][0]
        self.pixel_scale = pixel_scale

        image = torch.from_numpy(self.data["image"][0]).unsqueeze(0).unsqueeze(0)
        background = torch.from_numpy(self.data["background"][0]).unsqueeze(0).unsqueeze(0)
        self.image, self.background = apply_mask(
            image,
            background,
            regions=((1200, 1360, 1700, 1900), (280, 400, 1220, 1320)),
            mask_bg_val=865.0,
        )
        self.coadd_file = coadd_file

    def get_catalog(self, hlims, wlims):
        return CoaddFullCatalog.from_file(self.coadd_file, hlims, wlims)


class SimulatedFrame:
    def __init__(self, dataset: SimulatedDataset, n_tiles_h: int, n_tiles_w: int, cache_dir=None):
        dataset.to("cpu")
        if cache_dir is not None:
            sim_frame_path = Path(cache_dir) / "simulated_frame.pt"
        else:
            sim_frame_path = None
        if sim_frame_path and sim_frame_path.exists():
            tile_catalog, image, background = torch.load(sim_frame_path)
        else:
            print("INFO: started generating frame")
            tile_catalog = dataset.sample_prior(1, n_tiles_h, n_tiles_w)
            tile_catalog["galaxy_fluxes"] = dataset.image_decoder.get_galaxy_fluxes(
                tile_catalog["galaxy_bools"], tile_catalog["galaxy_params"]
            )
            image, background = dataset.simulate_image_from_catalog(tile_catalog)
            print("INFO: done generating frame")
            if sim_frame_path:
                torch.save((tile_catalog, image, background), sim_frame_path)

        self.tile_catalog = tile_catalog
        self.image = image
        self.background = background
        self.tile_slen = dataset.tile_slen
        self.bp = dataset.image_decoder.border_padding
        assert self.image.shape[0] == 1
        assert self.background.shape[0] == 1

    def get_catalog(self, hlims, wlims) -> FullCatalog:
        h, h_end = hlims[0] - self.bp, hlims[1] - self.bp
        w, w_end = wlims[0] - self.bp, wlims[1] - self.bp
        hlims_tile = int(np.floor(h / self.tile_slen)), int(np.ceil(h_end / self.tile_slen))
        wlims_tile = int(np.floor(w / self.tile_slen)), int(np.ceil(w_end / self.tile_slen))
        tile_dict = {}
        for k, v in self.tile_catalog.to_dict().items():
            tile_dict[k] = v[:, hlims_tile[0] : hlims_tile[1], wlims_tile[0] : wlims_tile[1]]
        tile_cat = TileCatalog(self.tile_slen, tile_dict)
        full_cat = tile_cat.to_full_params()
        full_cat["fluxes"] = (
            full_cat["galaxy_bools"] * full_cat["galaxy_fluxes"]
            + full_cat["star_bools"] * full_cat["fluxes"]
        )
        full_cat["mags"] = convert_flux_to_mag(full_cat["fluxes"])
        return full_cat


class SemiSyntheticFrame:
    def __init__(self, dataset: SimulatedDataset, coadd: str, n_tiles_h, n_tiles_w, cache_dir=None):
        dataset.to("cpu")
        self.bp = dataset.image_decoder.border_padding
        self.tile_slen = dataset.tile_slen
        self.coadd_file = coadd
        if cache_dir is not None:
            sim_frame_path = Path(cache_dir) / "simulated_frame.pt"
        else:
            sim_frame_path = None
        if sim_frame_path and sim_frame_path.exists():
            tile_catalog_dict, image, background = torch.load(sim_frame_path)
            tile_catalog = TileCatalog(self.tile_slen, tile_catalog_dict)
        else:
            hlim = (self.bp, self.bp + n_tiles_h * self.tile_slen)
            wlim = (self.bp, self.bp + n_tiles_w * self.tile_slen)
            full_coadd_cat = CoaddFullCatalog.from_file(coadd, hlim, wlim)
            if dataset.image_prior.galaxy_prior is not None:
                full_coadd_cat["galaxy_params"] = dataset.image_prior.galaxy_prior.sample(
                    full_coadd_cat.n_sources, "cpu"
                ).unsqueeze(0)
            full_coadd_cat.plocs = full_coadd_cat.plocs + 0.5
            max_sources = dataset.image_prior.max_sources
            tile_catalog = full_coadd_cat.to_tile_params(self.tile_slen, max_sources)
            tile_catalog["galaxy_fluxes"] = dataset.image_decoder.get_galaxy_fluxes(
                tile_catalog["galaxy_bools"], tile_catalog["galaxy_params"]
            )
            tile_catalog["star_bools"] = 1 - tile_catalog["galaxy_bools"]
            tile_catalog["fluxes"] = (
                tile_catalog["galaxy_bools"] * tile_catalog["galaxy_fluxes"]
                + tile_catalog["star_bools"] * tile_catalog["fluxes"]
            )
            tile_catalog["mags"] = convert_flux_to_mag(tile_catalog["fluxes"])
            fc = tile_catalog.to_full_params()
            assert fc.equals(
                full_coadd_cat, exclude=("galaxy_fluxes", "star_bools", "fluxes", "mags")
            )
            print("INFO: started generating frame")
            image, background = dataset.simulate_image_from_catalog(tile_catalog)
            print("INFO: done generating frame")
            if sim_frame_path:
                torch.save((tile_catalog.to_dict(), image, background), sim_frame_path)

        self.tile_catalog = tile_catalog
        self.image = image
        self.background = background
        assert self.image.shape[0] == 1
        assert self.background.shape[0] == 1

    def get_catalog(self, hlims, wlims):
        hlims = (hlims[0] - self.bp, hlims[1] - self.bp)
        wlims = (wlims[0] - self.bp, wlims[1] - self.bp)
        hlims_tile = (math.floor(hlims[0] / self.tile_slen), math.ceil(hlims[1] / self.tile_slen))
        wlims_tile = (math.floor(wlims[0] / self.tile_slen), math.ceil(wlims[1] / self.tile_slen))
        tile_catalog_cropped = self.tile_catalog.crop(hlims_tile, wlims_tile)
        full_catalog_cropped = tile_catalog_cropped.to_full_params()
        # Adjust for the fact that we cropped at the tile boundary
        full_catalog_cropped = full_catalog_cropped.crop(
            h_min=hlims[0] % self.tile_slen,
            h_max=-hlims[1] % self.tile_slen,
            w_min=wlims[0] % self.tile_slen,
            w_max=-wlims[1] % self.tile_slen,
        )
        full_catalog_cropped.plocs = full_catalog_cropped.plocs - 0.5
        return full_catalog_cropped


def apply_mask(image, background, regions, mask_bg_val=865.0):
    """Replaces specified regions with background noise."""
    for (h, h_end, w, w_end) in regions:
        img = image[:, :, h:h_end, w:w_end]
        image[:, :, h:h_end, w:w_end] = mask_bg_val + torch.tensor(
            mask_bg_val
        ).sqrt() * torch.randn_like(img)
        background[:, :, h:h_end, w:w_end] = mask_bg_val
    return image, background
