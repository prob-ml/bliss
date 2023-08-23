import random

import numpy as np
import torchvision as TF

from bliss.catalog import TileCatalog


def augment_data(tile_catalog, image, deconv_image=None):
    origin_tile = TileCatalog(4, tile_catalog)
    origin_full = origin_tile.to_full_params()

    num_transform_list = [1, 2, 3]
    num_transform = random.choices(num_transform_list, weights=[0.7, 0.2, 0.1], k=1)[0]
    transform_list = ["vflip", "hflip", "rotate90", "rotate180", "rotate270", "shift"]

    aug_method = np.random.choice(
        transform_list, num_transform, p=(0.1, 0.1, 0.1, 0.1, 0.1, 0.5), replace=False
    )

    aug_image, aug_tile, aug_deconv = origin_full, image, deconv_image
    for i in aug_method:
        if i == "vflip":
            aug_image, aug_tile, aug_deconv = aug_vflip(aug_image, aug_tile, aug_deconv)
        if i == "hflip":
            aug_image, aug_tile, aug_deconv = aug_hflip(aug_image, aug_tile, aug_deconv)
        if i == "rotate90":
            aug_image, aug_tile, aug_deconv = aug_rotate90(aug_image, aug_tile, aug_deconv)
        if i == "rotate180":
            aug_image, aug_tile, aug_deconv = aug_rotate180(aug_image, aug_tile, aug_deconv)
        if i == "rotate270":
            aug_image, aug_tile, aug_deconv = aug_rotate270(aug_image, aug_tile, aug_deconv)
        if i == "shift":
            aug_image, aug_tile, aug_deconv = aug_shift(aug_image, aug_tile, aug_deconv)

    return aug_image, aug_tile, aug_deconv


def aug_vflip(origin_full, image, deconv_image):
    aug_image = TF.transforms.functional.vflip(image)
    aug_deconv = None
    if deconv_image is not None:
        aug_deconv = TF.transforms.functional.vflip(deconv_image)
    image_size = image.size(2)
    origin_full["plocs"][:, :, 0] = image_size - origin_full["plocs"][:, :, 0] - 1
    aug_tile = origin_full.to_tile_params(4, 4).get_brightest_source_per_tile().to_dict()
    return aug_image, aug_tile, aug_deconv


def aug_hflip(origin_full, image, deconv_image):
    aug_image = TF.transforms.functional.hflip(image)
    aug_deconv = None
    if deconv_image is not None:
        aug_deconv = TF.transforms.functional.hflip(deconv_image)
    image_size = image.size(2)
    origin_full["plocs"][:, :, 1] = image_size - origin_full["plocs"][:, :, 1] - 1
    aug_tile = origin_full.to_tile_params(4, 4).get_brightest_source_per_tile().to_dict()
    return aug_image, aug_tile, aug_deconv


def aug_rotate90(origin_full, image, deconv_image):
    aug_image = TF.transforms.functional.rotate(image, 90)
    aug_deconv = None
    if deconv_image is not None:
        aug_deconv = TF.transforms.functional.rotate(deconv_image, 90)
    image_size = image.size(2)
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = plocs[:, :, 0]
    origin_full["plocs"][:, :, 0] = image_size - plocs[:, :, 1] - 1
    aug_tile = origin_full.to_tile_params(4, 4).get_brightest_source_per_tile().to_dict()
    return aug_image, aug_tile, aug_deconv


def aug_rotate180(origin_full, image, deconv_image):
    aug_image = TF.transforms.functional.rotate(image, 180)
    aug_deconv = None
    if deconv_image is not None:
        aug_deconv = TF.transforms.functional.rotate(deconv_image, 180)
    image_size = image.size(2)
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = image_size - plocs[:, :, 1] - 1
    origin_full["plocs"][:, :, 0] = image_size - plocs[:, :, 0] - 1
    aug_tile = origin_full.to_tile_params(4, 4).get_brightest_source_per_tile().to_dict()
    return aug_image, aug_tile, aug_deconv


def aug_rotate270(origin_full, image, deconv_image):
    aug_image = TF.transforms.functional.rotate(image, 270)
    aug_deconv = None
    if deconv_image is not None:
        aug_deconv = TF.transforms.functional.rotate(deconv_image, 270)
    image_size = image.size(2)
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = image_size - plocs[:, :, 0] - 1
    origin_full["plocs"][:, :, 0] = plocs[:, :, 1]
    aug_tile = origin_full.to_tile_params(4, 4).get_brightest_source_per_tile().to_dict()
    return aug_image, aug_tile, aug_deconv


def aug_shift(origin_full, image, deconv_image):
    shift_x = random.randint(0, 3)
    shift_y = random.randint(0, 3)
    shift_xy = (shift_x, shift_y)
    image_size = image.size(2)
    aug_deconv = None
    pad_image = TF.transforms.functional.pad(image, shift_xy, padding_mode="reflect")
    if deconv_image is not None:
        pad_deconv = TF.transforms.functional.pad(deconv_image, shift_xy, padding_mode="reflect")
        aug_deconv = pad_deconv[:, :, :image_size, :image_size]

    aug_image = pad_image[:, :, :image_size, :image_size]
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = plocs[:, :, 1] + shift_x
    origin_full["plocs"][:, :, 0] = plocs[:, :, 0] + shift_y

    aug_tile = origin_full.to_tile_params(4, 4).get_brightest_source_per_tile().to_dict()
    return aug_image, aug_tile, aug_deconv
