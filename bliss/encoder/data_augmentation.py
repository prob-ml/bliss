import random

import torch
from einops import rearrange
from torchvision.transforms import functional as TF

from bliss.catalog import TileCatalog


def augment_batch(batch):
    original_images = [batch["images"], batch["background"]]
    if "deconvolution" in batch:
        original_images.append(batch["background"])
    original_image_stack = torch.stack(original_images, dim=2)

    aug_image_stack, aug_tile = augment_data(batch["tile_catalog"], original_image_stack)

    batch["tile_catalog"] = aug_tile

    batch["images"] = aug_image_stack[:, :, 0, :, :]
    batch["background"] = aug_image_stack[:, :, 1, :, :]
    if "deconvolution" in batch:
        batch["deconvolution"] = aug_image_stack[:, :, 2, :, :]  # noqa: WPS529


def augment_data(tile_catalog, image):
    origin_tile = TileCatalog(4, tile_catalog)
    origin_full = origin_tile.to_full_catalog()
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
    aug_tile = (
        aug_full.to_tile_catalog(4, 4, filter_oob=True).get_brightest_sources_per_tile().to_dict()
    )
    return aug_image, aug_tile


def aug_vflip(origin_full, image):
    aug_image = TF.vflip(image)
    image_size = image.size(3)
    origin_full["plocs"][:, :, 0] = image_size - origin_full["plocs"][:, :, 0] - 1
    return aug_image, origin_full


def aug_rotate90(origin_full, image):
    aug_image = rotate_images(image, 90)
    image_size = image.size(3)
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = plocs[:, :, 0]
    origin_full["plocs"][:, :, 0] = image_size - plocs[:, :, 1] - 1
    return aug_image, origin_full


def aug_rotate180(origin_full, image):
    aug_image = rotate_images(image, 180)
    image_size = image.size(3)
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = image_size - plocs[:, :, 1] - 1
    origin_full["plocs"][:, :, 0] = image_size - plocs[:, :, 0] - 1
    return aug_image, origin_full


def aug_rotate270(origin_full, image):
    aug_image = rotate_images(image, 270)
    image_size = image.size(3)
    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = image_size - plocs[:, :, 0] - 1
    origin_full["plocs"][:, :, 0] = plocs[:, :, 1]
    return aug_image, origin_full


def rotate_images(image, degree):
    num_batch = image.size(0)
    combined_image = rearrange(image, "bt bd ch h w -> (bt bd) ch h w")
    rotated_image = TF.rotate(combined_image, degree)
    return rearrange(rotated_image, "(bt bd) ch h w -> bt bd ch h w", bt=num_batch)


def aug_shift(origin_full, image):
    shift_x = random.randint(-1, 2)
    shift_y = random.randint(-1, 2)
    image_size = image.size(3)
    image_lim = [2 - shift_x, 2 - shift_x + image_size, 2 - shift_y, 2 - shift_y + image_size]

    num_batch = image.size(0)
    combined_image = rearrange(image, "bt bd ch h w -> (bt bd) ch h w")
    pad_combined_image = TF.pad(combined_image, (2, 2), padding_mode="reflect")
    pad_image = rearrange(pad_combined_image, "(bt bd) ch h w -> bt bd ch h w", bt=num_batch)
    aug_image = pad_image[:, :, :, image_lim[0] : image_lim[1], image_lim[2] : image_lim[3]]

    plocs = origin_full["plocs"].clone()
    origin_full["plocs"][:, :, 1] = plocs[:, :, 1] + shift_y
    origin_full["plocs"][:, :, 0] = plocs[:, :, 0] + shift_x

    return aug_image, origin_full
