import numpy as np
import torch
from einops import rearrange
from torch.distributions import Normal
from tqdm import tqdm

from bliss.catalog import TileCatalog, get_images_in_tiles
from bliss.inference import get_padded_scene
from bliss.models.decoder import ImageDecoder


def iterative_optimization(
    tile_catalog: TileCatalog,
    decoder: ImageDecoder,
    image, background,
    h_range, w_range,
    source_prob = 0.004,
    #threshold = 0.99,
    #threshold = 0.5,
    threshold = 0.5,
    ):
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
            h_idx = int(source_idx) // n_tiles_w
            w_idx = int(source_idx) % n_tiles_w
            assert tile_catalog.is_on_array[0, h_idx, w_idx] > 0
            # Get subset of tile catalog to render
            h_min = max(0, h_idx - 10)
            h_max = min(n_tiles_h, h_idx + 11)
            hlims = h_min, h_max
            h_idx_cropped = min(10, h_idx - h_min)

            w_min = max(0, w_idx - 10)
            w_max = min(n_tiles_w, w_idx + 11)
            wlims = w_min, w_max
            w_idx_cropped = min(10, w_idx - w_min)

            subset = tile_catalog.crop(hlims, wlims).continguous()
            rendered_ptiles = decoder._render_ptiles(subset)
            
            rendered_ptile = rendered_ptiles[0, h_idx_cropped, w_idx_cropped]

            ptile = padded_tiles[0, h_idx, w_idx]
            img = ptile[0]
            #bg = ptile[1]
            #bg = 900.0
            #bg = 865.0
            bg = 870.0
            # bg = 887.0
            # bg = 873.6922

            ll = Normal(rendered_ptile + bg, (rendered_ptile + bg).sqrt()).log_prob(img).sum()

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
            rendered_ptile_ctft = rendered_ptiles_ctft[0, h_idx_cropped, w_idx_cropped]
            ll_counterfact = Normal(rendered_ptile_ctft + bg, (rendered_ptile_ctft + bg).sqrt()).log_prob(img).sum()

            def debug_fig(img):
                from matplotlib import pyplot as plt
                plt.imshow(img)
                plt.savefig("debug.png")

            # if h_idx == 42:
            #     print("here")
            #     pass

            post = ll + torch.log(torch.tensor(source_prob))
            post_counterfact = ll_counterfact + torch.log(torch.tensor(1 - source_prob))
            log_prob = post.item() - np.logaddexp(post.item(), post_counterfact.item())

            flux = rendered_ptile.sum()
            if flux < 622:
                print("here b/c flux")
            if log_prob < np.log(threshold):
                if log_prob > np.log(0.9):
                    print("here")
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
