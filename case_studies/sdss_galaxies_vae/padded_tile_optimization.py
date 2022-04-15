import torch
from einops import rearrange
from torch.distributions import Normal
from bliss.catalog import TileCatalog, get_images_in_tiles
from bliss.models.decoder import ImageDecoder


def iterative_optimization(tile_catalog: TileCatalog, decoder: ImageDecoder, image, background):
    assert tile_catalog.batch_size == 1
    _, n_tiles_h, n_tiles_w, n_sources_per_tile, _ = tile_catalog.locs.shape
    padded_tiles = get_images_in_tiles(
        torch.cat((image, background), dim=1), decoder.tile_slen, decoder.ptile_slen
    )
    indices_to_retrieve, _ = tile_catalog._get_indices_of_on_sources()
    on_tiles = rearrange(indices_to_retrieve, "1 ns -> ns")
    for source_idx in on_tiles:
        h_idx = int(source_idx) // n_tiles_h
        w_idx = int(source_idx) % n_tiles_h
        # Get subset of tile catalog to render
        subset = tile_catalog.crop((h_idx - 10, h_idx + 10), (w_idx - 10, w_idx + 10))
        rendered_ptiles = decoder._render_ptiles(subset)
        rendered_ptile = rendered_ptiles[0, 10, 10]
        ptile = padded_tiles[0, h_idx, w_idx]
        ll = Normal(rendered_ptile + ptile[1], (rendered_ptile + ptile[1]).sqrt()).log_prob(
            ptile[0]
        )

        tile_catalog_counterfact = tile_catalog.copy()
        n_sources_counterfact = tile_catalog_counterfact.n_sources.clone()
        n_sources_counterfact[0, h_idx, w_idx] = 1 - n_sources_counterfact[0, h_idx, w_idx]
        tile_catalog_counterfact.n_sources = n_sources_counterfact
        subset = tile_catalog_counterfact.crop((h_idx - 10, h_idx + 10), (w_idx - 10, w_idx + 10))
        rendered_ptiles = decoder._render_ptiles(subset)
        rendered_ptile = rendered_ptiles[0, 10, 10]
        ll = Normal(rendered_ptile + ptile[1], (rendered_ptile + ptile[1]).sqrt()).log_prob(
            ptile[0]
        )

        if ll_counterfact > ll:
            switch_source_and_update_catalog()


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
