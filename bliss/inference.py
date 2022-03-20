from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from einops import rearrange, reduce
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from bliss.catalog import TileCatalog
from bliss.datasets.sdss import SloanDigitalSkySurvey, convert_flux_to_mag
from bliss.datasets.simulated import SimulatedDataset
from bliss.encoder import Encoder
from bliss.models.decoder import ImageDecoder
from bliss.reporting import CoaddFullCatalog


def reconstruct_scene_at_coordinates(
    encoder: Encoder,
    decoder: ImageDecoder,
    img: Tensor,
    background: Tensor,
    h_range: Tuple[int, int],
    w_range: Tuple[int, int],
    slen: int = 300,
    device=None,
) -> Tuple[Tensor, TileCatalog]:
    """Reconstruct all objects contained within a scene, padding as needed.

    This function will run the encoder and decoder on a padded image containing the image specified
    via h, w, and scene_length, to ensure that all objects can be detected and reconstructed.

    Args:
        encoder:
            Trained Encoder module.
        decoder:
            Trained ImageDecoder module.
        img:
            A NxCxHxW tensor of N HxW images (with C bands).
        background:
            A NxCxHxW tensor of N HxW background values (with C bands).
        h_range:
            Range of height coordinates.
        w_range:
            Range of width coordinates.
        slen:
            The side-lengths of smaller chunks to create. Defaults to 80.
        device:
            Device used for rendering each chunk (i.e. a torch.device). Note
            that chunks are moved onto and off the device to allow for rendering
            larger images.

    Returns:
        A tuple of two items:
        -  recon_at_coords: A NxCxHxW Tensor of the reconstructed images
            at the coordinates specified hy h, w, and scene_length.
        -  map_scene: The maximum-a-posteriori catalog estimated from the image
            (and surrounding border padding). Note that this may contain objects detected
            outside the bounds of the image given, but their coordinates will be relative to (h, w).
            In other words, the locations of these out-of-bounds objects will either be negative if
            they are to the left/above (h, w) or greater than scene_length.

    """
    bp = encoder.border_padding
    h_range_pad = (h_range[0] - bp, h_range[1] + bp)
    w_range_pad = (w_range[0] - bp, w_range[1] + bp)

    # First get the mininum coordinates to ensure everything is detected
    scene = img[:, :, h_range_pad[0] : h_range_pad[1], w_range_pad[0] : w_range_pad[1]]
    bg_scene = background[:, :, h_range_pad[0] : h_range_pad[1], w_range_pad[0] : w_range_pad[1]]
    assert scene.shape[2] == h_range_pad[1] - h_range_pad[0]
    assert scene.shape[3] == w_range_pad[1] - w_range_pad[0]
    chunked_scene = ChunkedScene(scene, bg_scene, slen, bp)
    recon, tile_map_scene = chunked_scene.reconstruct(encoder, decoder, device)
    assert recon.shape == scene.shape
    recon += bg_scene
    # Get reconstruction at coordinates
    recon_at_coords = recon[
        :,
        :,
        bp:-bp,
        bp:-bp,
    ]
    return recon_at_coords, tile_map_scene


class ChunkedScene:
    def __init__(self, scene: Tensor, bg_scene: Tensor, slen: int, bp: int):
        """Split scenes into square chunks of side length `slen+bp*2` using `F.unfold`."""
        self.slen = slen
        self.bp = bp
        kernel_size = slen + bp * 2
        self.kernel_size = kernel_size
        self.output_size = (scene.shape[2], scene.shape[3])
        self.chunk_dict = {}
        self.bg_dict = {}

        n_chunks_h = (scene.shape[2] - (bp * 2)) // slen
        n_chunks_w = (scene.shape[3] - (bp * 2)) // slen
        self.n_chunks_h_main = n_chunks_h
        self.n_chunks_w_main = n_chunks_w
        self.chunk_dict["main"] = self._chunk_image(scene, (kernel_size, kernel_size), slen)
        self.bg_dict["main"] = self._chunk_image(bg_scene, (kernel_size, kernel_size), slen)

        # Get leftover chunks
        bottom_border_start = bp + slen * n_chunks_h - bp
        bottom_chunk_height = scene.shape[2] - bottom_border_start
        assert bottom_chunk_height >= bp * 2

        right_border_start = bp + slen * n_chunks_w - bp
        right_chunk_width = scene.shape[3] - right_border_start
        assert right_chunk_width >= bp * 2

        if bottom_chunk_height > bp * 2:
            bottom_border = scene[:, :, bottom_border_start:, : (right_border_start + 2 * bp)]
            bg_bottom_border = bg_scene[:, :, bottom_border_start:, : (right_border_start + 2 * bp)]
            self.chunk_dict["bottom"] = self._chunk_image(
                bottom_border, (bottom_chunk_height, kernel_size), slen
            )
            self.bg_dict["bottom"] = self._chunk_image(
                bg_bottom_border, (bottom_chunk_height, kernel_size), slen
            )

        if right_chunk_width > bp * 2:
            right_border = scene[:, :, : (bottom_border_start + 2 * bp), right_border_start:]
            bg_right_border = bg_scene[:, :, : (bottom_border_start + 2 * bp), right_border_start:]
            self.chunk_dict["right"] = self._chunk_image(
                right_border, (kernel_size, right_chunk_width), slen
            )
            self.bg_dict["right"] = self._chunk_image(
                bg_right_border, (kernel_size, right_chunk_width), slen
            )

        if (bottom_chunk_height > bp * 2) and (right_chunk_width > bp * 2):
            bottom_right_border = scene[:, :, bottom_border_start:, right_border_start:]
            bg_bottom_right_border = bg_scene[:, :, bottom_border_start:, right_border_start:]
            self.chunk_dict["bottom_right"] = bottom_right_border
            self.bg_dict["bottom_right"] = bg_bottom_right_border

    def _chunk_image(self, image, kernel_size, stride):
        chunks = F.unfold(image, kernel_size=kernel_size, stride=stride)
        return rearrange(
            chunks,
            "b (c h w) n -> (b n) c h w",
            c=image.shape[1],
            h=kernel_size[0],
            w=kernel_size[1],
        )

    def reconstruct(self, encoder, decoder, device):
        chunk_est_dict = {}
        for chunk_type, chunks in self.chunk_dict.items():
            bgs = self.bg_dict[chunk_type]
            chunk_est_dict[chunk_type] = self._reconstruct_chunks(
                chunks, bgs, encoder, decoder, device
            )
        scene_recon = self._combine_into_scene(chunk_est_dict)
        chunk_tile_maps_dict = {k: v["tile_maps"] for k, v in chunk_est_dict.items()}
        tile_map_recon = self._combine_tile_maps(chunk_tile_maps_dict)
        return scene_recon, tile_map_recon

    def _reconstruct_chunks(self, chunks, bgs, encoder, decoder, device):
        reconstructions = []
        tile_maps = []
        for chunk, bg in tqdm(zip(chunks, bgs), desc="Reconstructing chunks"):
            recon, tile_map = reconstruct_img(
                encoder, decoder, chunk.unsqueeze(0).to(device), bg.unsqueeze(0).to(device)
            )
            reconstructions.append(recon.cpu())
            tile_maps.append(tile_map.cpu())
        return {
            "reconstructions": torch.cat(reconstructions, dim=0),
            "tile_maps": tile_maps,
        }

    def _combine_into_scene(self, chunk_est_dict: Dict):
        main = chunk_est_dict["main"]["reconstructions"]
        main = rearrange(
            main,
            "(nch ncw) c h w -> nch ncw c h w",
            nch=self.n_chunks_h_main,
            ncw=self.n_chunks_w_main,
        )

        right = chunk_est_dict.get("right")
        if right is not None:
            right = right["reconstructions"]
            right_padding = self.kernel_size - right.shape[-1]
            right = F.pad(right, (0, right_padding, 0, 0))
            right = rearrange(right, "nch c h w -> nch 1 c h w")
            main = torch.cat((main, right), dim=1)
        else:
            right_padding = 0

        bottom = chunk_est_dict.get("bottom")
        if bottom is not None:
            bottom = bottom["reconstructions"]
            bottom_padding = self.kernel_size - bottom.shape[-2]
            bottom = F.pad(bottom, (0, 0, 0, bottom_padding))

            bottom_right = chunk_est_dict.get("bottom_right")
            if bottom_right is not None:
                bottom_right = bottom_right["reconstructions"]
                bottom_right = F.pad(bottom_right, (0, right_padding, 0, bottom_padding))
                bottom = torch.cat((bottom, bottom_right), dim=0)
            bottom = rearrange(bottom, "ncw c h w -> 1 ncw c h w")
            main = torch.cat((main, bottom), axis=0)
        else:
            bottom_padding = 0
        image_flat = rearrange(main, "nch ncw c h w -> 1 (c h w) (nch ncw)")
        output_size = (self.output_size[0] + bottom_padding, self.output_size[1] + right_padding)
        image = F.fold(
            image_flat, output_size=output_size, kernel_size=self.kernel_size, stride=self.slen
        )
        return image[
            :,
            :,
            : (-bottom_padding if bottom_padding else None),
            : (-right_padding if right_padding else None),
        ]

    def _combine_tile_maps(self, chunk_tile_maps_dict: Dict[str, List[TileCatalog]]):
        main = self._tile_catalog_array(chunk_tile_maps_dict["main"])
        n_chunks_h_main = self.n_chunks_h_main
        n_chunks_w_main = self.n_chunks_w_main

        main = main.reshape(n_chunks_h_main, n_chunks_w_main)

        right = chunk_tile_maps_dict.get("right")
        if right is not None:
            right = self._tile_catalog_array(right).reshape(-1, 1)
            main = np.concatenate((main, right), axis=1)

        bottom = chunk_tile_maps_dict.get("bottom")
        if bottom is not None:
            bottom_right = chunk_tile_maps_dict.get("bottom_right")
            if bottom_right is not None:
                bottom += bottom_right
            bottom = self._tile_catalog_array(bottom).reshape(1, -1)
            main = np.concatenate((main, bottom), axis=0)

        tile_map_list = []
        for tile_map_row in main:
            tile_map_row_combined = cat_tile_catalog(tile_map_row, 1)
            tile_map_list.append(tile_map_row_combined)
        return cat_tile_catalog(tile_map_list, 0)

    @staticmethod
    def _tile_catalog_array(tile_catalogs: List[TileCatalog]):
        out = np.zeros(len(tile_catalogs), dtype=TileCatalog)
        for i, tile_catalog in enumerate(tile_catalogs):
            out[i] = tile_catalog
        return out


def reconstruct_img(
    encoder: Encoder, decoder: ImageDecoder, img: Tensor, bg: Tensor
) -> Tuple[Tensor, TileCatalog]:

    with torch.no_grad():
        tile_map = encoder.max_a_post(img, bg)
        recon_image = decoder.render_images(tile_map)
        tile_map["galaxy_fluxes"] = decoder.get_galaxy_fluxes(
            tile_map["galaxy_bools"], tile_map["galaxy_params"]
        )
    return recon_image, tile_map


def cat_tile_catalog(tile_catalogs: Sequence[TileCatalog], tile_dim: int = 0) -> TileCatalog:
    assert tile_dim in {0, 1}
    out = {}
    tile_catalog_dicts = [tm.to_dict() for tm in tile_catalogs]
    for k in tile_catalog_dicts[0].keys():
        tensors = [tm[k] for tm in tile_catalog_dicts]
        value = torch.cat(tensors, dim=(tile_dim + 1))
        out[k] = value
    return TileCatalog(tile_catalogs[0].tile_slen, out)


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

    def get_catalog(self, hlims, wlims):
        h, h_end = hlims[0] - self.bp, hlims[1] - self.bp
        w, w_end = wlims[0] - self.bp, wlims[1] - self.bp
        hlims_tile = int(np.floor(h / self.tile_slen)), int(np.ceil(h_end / self.tile_slen))
        wlims_tile = int(np.floor(w / self.tile_slen)), int(np.ceil(w_end / self.tile_slen))
        tile_cat_cropped = {}
        for k, v in self.tile_catalog.to_dict().items():
            tile_cat_cropped[k] = v[:, hlims_tile[0] : hlims_tile[1], wlims_tile[0] : wlims_tile[1]]
        tile_cat_cropped = TileCatalog(self.tile_slen, tile_cat_cropped)
        full_cat = tile_cat_cropped.to_full_params()
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
            full_coadd_cat["galaxy_params"] = (
                torch.randn((1, full_coadd_cat.n_sources, 32)) * full_coadd_cat["galaxy_bools"]
            )
            full_coadd_cat.plocs = full_coadd_cat.plocs + 0.5
            max_sources = dataset.image_prior.max_sources
            tile_catalog = full_coadd_cat.to_tile_params(self.tile_slen, max_sources)
            tile_catalog["galaxy_fluxes"] = dataset.image_decoder.get_galaxy_fluxes(
                tile_catalog["galaxy_bools"], tile_catalog["galaxy_params"]
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
        h, h_end = hlims[0] - self.bp, hlims[1] - self.bp
        w, w_end = wlims[0] - self.bp, wlims[1] - self.bp
        hlims_tile = int(np.floor(h / self.tile_slen)), int(np.ceil(h_end / self.tile_slen))
        wlims_tile = int(np.floor(w / self.tile_slen)), int(np.ceil(w_end / self.tile_slen))
        # tile_cat_cropped = {}
        # for k, v in self.tile_catalog.items():
        #     tile_cat_cropped[k] = v[:, hlims_tile[0] : hlims_tile[1], wlims_tile[0] : wlims_tile[1]]
        tile_cat_cropped = self.tile_catalog.crop(hlims_tile, wlims_tile)
        full_cat = tile_cat_cropped.to_full_params()
        full_cat["star_bools"] = 1 - full_cat["galaxy_bools"]
        full_cat["fluxes"] = (
            full_cat["galaxy_bools"] * full_cat["galaxy_fluxes"]
            + full_cat["star_bools"] * full_cat["fluxes"]
        )
        full_cat["mags"] = convert_flux_to_mag(full_cat["fluxes"])
        full_cat.plocs = full_cat.plocs - 1.0
        return full_cat


def apply_mask(image, background, regions, mask_bg_val=865.0):
    """Replaces specified regions with background noise."""
    for (h, h_end, w, w_end) in regions:
        img = image[:, :, h:h_end, w:w_end]
        image[:, :, h:h_end, w:w_end] = mask_bg_val + torch.tensor(
            mask_bg_val
        ).sqrt() * torch.randn_like(img)
        background[:, :, h:h_end, w:w_end] = mask_bg_val
    return image, background


def infer_blends(tile_map: TileCatalog, tile_range: int) -> Tensor:
    n_galaxies_per_tile = reduce(tile_map["galaxy_bools"], "n nth ntw s 1 -> n 1 nth ntw", "sum")
    kernel = torch.ones((1, 1, tile_range, tile_range))
    blends = F.conv2d(n_galaxies_per_tile, kernel)
    # Pad output with zeros
    output = torch.zeros_like(n_galaxies_per_tile)
    output[:, :, (tile_range - 1) :, (tile_range - 1) :] += blends
    return rearrange(output, "n 1 nth ntw -> n nth ntw 1 1")
