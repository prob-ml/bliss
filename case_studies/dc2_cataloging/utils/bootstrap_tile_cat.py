import copy

import numpy as np

from bliss.catalog import TileCatalog


def bootstrap_tile_cat(ori_tile_cat: TileCatalog, seed: int):
    batch_size = ori_tile_cat.batch_size
    n_tiles_h = ori_tile_cat.n_tiles_h
    n_tiles_w = ori_tile_cat.n_tiles_w
    tile_dict = copy.copy(ori_tile_cat.data)

    rng = np.random.default_rng(seed)

    for i in range(batch_size):
        random_indices = rng.choice(n_tiles_h * n_tiles_w, (n_tiles_h * n_tiles_w,), replace=True)
        for k, v in tile_dict.items():
            cur_batch_v = v[i].flatten(0, 1)
            cur_batch_v = cur_batch_v[random_indices.tolist()]
            tile_dict[k][i] = cur_batch_v.view(n_tiles_h, n_tiles_w, *cur_batch_v.shape[1:])

    return TileCatalog(tile_dict)
