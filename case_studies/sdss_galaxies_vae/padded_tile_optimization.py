import torch

from bliss.catalog import TileCatalog
from bliss.models.decoder import ImageDecoder
def padded_tile_optimization(tile_catalog: TileCatalog, decoder: ImageDecoder):
    # Flaws
    # - Not accounting for dependence
    #     - Should update the global n_sources after each loop
    assert tile_catalog.batch_size == 1
    best = torch.zeros_like(tile_catalog.n_sources)
    for coord in ((0,0), (0, 1), (1, 0), (1, 1)):
        padded_tiles = get_image_in_padded_tiles(image, background)
        padded_tiles_0_0 = get_padded_tiles_0_0(padded_tiles)
        tile_catalog_0_0 = get_tile_catalog_0_0(tile_catalog)
        tile_catalog_0_0_counterfactual = get_tile_catalog_0_0_counterfactual(tile_catalog)
        rendered_tiles_0_0 = decoder._render_ptiles(tile_catalog_0_0)
        rendered_tiles_0_0_counterfactual = decoder._render_ptiles(tile_catalog_0_0_counterfactual)
        log_liks_0_0 = get_log_liks(padded_tiles_0_0, rendered_tiles_0_0)
        log_liks_0_0_counterfactual = get_log_liks(padded_tiles_0_0, rendered_tiles_0_0_counterfactual)
        logs_liks_0_0_stacked = torch.stack((log_liks_0_0, log_liks_0_0_counterfactual), dim=-1)
        best_0_0 = logs_liks_0_0_stacked.argmin(-1)
        best[0, 0::2, 0::2] = best_0_0



