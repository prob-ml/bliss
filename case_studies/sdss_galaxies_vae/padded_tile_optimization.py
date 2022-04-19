import torch
from einops import rearrange
from torch.distributions import Normal
from tqdm import tqdm

from bliss.catalog import TileCatalog, get_images_in_tiles
from bliss.inference import get_padded_scene
from bliss.models.decoder import ImageDecoder


def iterative_optimization(tile_catalog: TileCatalog, decoder: ImageDecoder, image, background, h_range, w_range):
    with torch.no_grad():
        device = decoder.device
        tile_catalog = tile_catalog.copy().to(device)
        assert tile_catalog.batch_size == 1
        _, n_tiles_h, n_tiles_w, n_sources_per_tile, _ = tile_catalog.locs.shape
        image, background = get_padded_scene(image, background, h_range, w_range, decoder.border_padding)
        padded_tiles = get_images_in_tiles(
            torch.cat((image, background), dim=1), decoder.tile_slen, decoder.ptile_slen
        ).to(device)
        indices_to_retrieve, _ = tile_catalog._get_indices_of_on_sources()
        on_tiles = rearrange(indices_to_retrieve, "1 ns -> ns")
        n_changes = 0
        for source_idx in tqdm(on_tiles):
            h_idx = int(source_idx) // n_tiles_h
            w_idx = int(source_idx) % n_tiles_h
            # Get subset of tile catalog to render
            hlims = max(0, h_idx - 10), min(n_tiles_h, h_idx + 10)
            wlims = max(0, w_idx - 10), min(n_tiles_w, w_idx + 10)
            subset = tile_catalog.crop(hlims, wlims).continguous()
            rendered_ptiles = decoder._render_ptiles(subset)
            rendered_ptile = rendered_ptiles[0, min(10, h_idx), min(10, w_idx)]
            ptile = padded_tiles[0, h_idx, w_idx]
            ll = Normal(rendered_ptile + ptile[1], (rendered_ptile + ptile[1]).sqrt()).log_prob(
                ptile[0]
            ).sum()

            tile_catalog_counterfact = tile_catalog.copy()
            n_sources_counterfact = tile_catalog_counterfact.n_sources.clone()
            galaxy_bools = tile_catalog_counterfact["galaxy_bools"].clone()
            star_bools = tile_catalog_counterfact["star_bools"].clone()
            n_sources_counterfact[0, h_idx, w_idx] = 1 - n_sources_counterfact[0, h_idx, w_idx]
            galaxy_bools[0, h_idx, w_idx] = 0.0
            star_bools[0, h_idx, w_idx] = 0.0
            tile_catalog_counterfact.n_sources = n_sources_counterfact
            tile_catalog_counterfact["galaxy_bools"] = galaxy_bools
            tile_catalog_counterfact["star_bools"] = star_bools
            subset_counterfact = tile_catalog_counterfact.crop(hlims, wlims).continguous()
            rendered_ptiles_ctft = decoder._render_ptiles(subset_counterfact)
            rendered_ptile_ctft = rendered_ptiles_ctft[0, min(10, h_idx), min(10, w_idx)]
            ll_counterfact = Normal(rendered_ptile_ctft + ptile[1], (rendered_ptile_ctft + ptile[1]).sqrt()).log_prob(
                ptile[0]
            ).sum()

            if h_idx == 42:
                print("here")
                pass

            if ll_counterfact > ll:
                tile_catalog = tile_catalog_counterfact
                # tile_catalog.n_sources = n_sources_counterfact
                # tile_cal
                n_changes += 1
        print(f"N changed: {n_changes}")
        return tile_catalog.to("cpu")

## Checkerboard
def padded_tile_optimization(tile_catalog: TileCatalog, decoder: ImageDecoder):
    # Flaws
    # - Not accounting for dependence
    #     - Should update the global n_sources after each loop
    assert tile_catalog.batch_size == 1
    best = torch.zeros_like(tile_catalog.n_sources)
    for coord in ((0, 0), (0, 1), (1, 0), (1, 1)):
        padded_tiles = get_image_in_padded_tiles(image, background)
        padded_tiles_0_0 = get_padded_tiles_0_0(padded_tiles)
        tile_catalog_0_0 = get_tile_catalog_0_0(tile_catalog)
        tile_catalog_0_0_counterfactual = get_tile_catalog_0_0_counterfactual(tile_catalog)
        rendered_tiles_0_0 = decoder._render_ptiles(tile_catalog_0_0)
        rendered_tiles_0_0_counterfactual = decoder._render_ptiles(tile_catalog_0_0_counterfactual)
        log_liks_0_0 = get_log_liks(padded_tiles_0_0, rendered_tiles_0_0)
        log_liks_0_0_counterfactual = get_log_liks(
            padded_tiles_0_0, rendered_tiles_0_0_counterfactual
        )
        logs_liks_0_0_stacked = torch.stack((log_liks_0_0, log_liks_0_0_counterfactual), dim=-1)
        best_0_0 = logs_liks_0_0_stacked.argmin(-1)
        best[0, 0::2, 0::2] = best_0_0
