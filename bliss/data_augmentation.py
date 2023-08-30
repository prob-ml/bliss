import random

import torch
from torchvision.transforms import functional as TF

from bliss.catalog import TileCatalog


def augment_data(tile_catalog, image):
    origin_tile = TileCatalog(4, tile_catalog)
    origin_full = origin_tile.to_full_params()
    aug_full, aug_image = origin_full, image

    rotate_list = [None, aug_rotate90, aug_rotate180, aug_rotate270]
    flip_list = [None, aug_vflip]
    rotate_choice = random.choice(rotate_list)
    flip_choice = random.choice(flip_list)

    if rotate_choice is not None:
        aug_image, aug_full = rotate_choice(aug_full, aug_image)
    if flip_choice is not None:
        aug_image, aug_full = flip_choice(aug_full, aug_image)

    aug_image, aug_full = aug_shift(aug_full, aug_image)
    aug_tile = aug_full.to_tile_params(4, 4).get_brightest_source_per_tile().to_dict()
    return aug_image, aug_tile


def aug_vflip(origin_full, image):
    aug_image = TF.vflip(image)
    image_size = image.size(3)
    origin_full["plocs"][:, :, 0] = image_size - origin_full["plocs"][:, :, 0] - 1
    return aug_image, origin_full


def aug_rotate90(origin_full, image):
    num_channel = image.size(2)
    aug_images = []
    for i in range(num_channel):
        rotated_slice = TF.rotate(image[:, :, i, :, :], 90)
        aug_images.append(rotated_slice)
    aug_image = torch.stack(aug_images, dim=2)

    image_size = image.size(3)
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = plocs[:, :, 0]
    origin_full["plocs"][:, :, 0] = image_size - plocs[:, :, 1] - 1
    return aug_image, origin_full


def aug_rotate180(origin_full, image):
    num_channel = image.size(2)
    aug_images = []
    for i in range(num_channel):
        rotated_slice = TF.rotate(image[:, :, i, :, :], 180)
        aug_images.append(rotated_slice)
    aug_image = torch.stack(aug_images, dim=2)

    image_size = image.size(3)
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = image_size - plocs[:, :, 1] - 1
    origin_full["plocs"][:, :, 0] = image_size - plocs[:, :, 0] - 1
    return aug_image, origin_full


def aug_rotate270(origin_full, image):
    num_channel = image.size(2)
    aug_images = []
    for i in range(num_channel):
        rotated_slice = TF.rotate(image[:, :, i, :, :], 270)
        aug_images.append(rotated_slice)
    aug_image = torch.stack(aug_images, dim=2)

    image_size = image.size(3)
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = image_size - plocs[:, :, 0] - 1
    origin_full["plocs"][:, :, 0] = plocs[:, :, 1]
    return aug_image, origin_full


def aug_shift(origin_full, image):
    shift_x = random.randint(-1, 2)
    shift_y = random.randint(-1, 2)
    image_size = image.size(3)
    image_lim = [2 - shift_x, 2 - shift_x + image_size, 2 - shift_y, 2 - shift_y + image_size]

    num_channel = image.size(2)
    aug_images = []
    for i in range(num_channel):
        pad_slice = TF.pad(image[:, :, i, :, :], (2, 2), padding_mode="reflect")
        aug_images.append(pad_slice[:, :, image_lim[0] : image_lim[1], image_lim[2] : image_lim[3]])
    aug_image = torch.stack(aug_images, dim=2)

    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = plocs[:, :, 1] + shift_x
    origin_full["plocs"][:, :, 0] = plocs[:, :, 0] + shift_y

    return aug_image, origin_full
