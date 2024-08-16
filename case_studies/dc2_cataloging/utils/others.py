from pytorch_lightning.utilities import move_data_to_device

from bliss.catalog import TileCatalog


def move_tile_cat_to_device(ori_tile_cat: TileCatalog, device):
    tile_dict = move_data_to_device(ori_tile_cat.data, device=device)
    return TileCatalog(tile_dict)
