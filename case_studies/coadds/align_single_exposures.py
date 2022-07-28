import galsim
import numpy as np
import torch
import torch.nn.functional as F

from bliss.models.decoder import get_mgrid


def align_single_exposures(img0, images, slen, dithers, scale=0.393):
    """Aligns multiple single exposure images that are dithered by some sub-pixel amount.
    Args:
        img0: Tensor of shape `(1 x C x H x W)` C is the number of band, H is height and W is weight,
            containing image data.
        images: Tensor of shape `(N x C x H x W)` where N is the number of dithers (len(dithers)),
            C is the number of band, H is height and W is weight, containing image data.
        slen: size of images (H and W)
        dithers: List of pairs of sub-pixel amounts that img0 is shifted by in x and y directions
        scale: pixel_scale
    Returns:
        Tensor of shape `(N x H-2 x W-2)`. Aligned images with 1 pixel cropped from from each
            direction of height and width.
    """
    img0 = galsim.Image(np.array(img0), wcs=galsim.PixelScale(scale))
    wcs0 = img0.wcs

    sgrid = (get_mgrid(slen) - (-1)) / 2 * (slen)
    sgrid = sgrid.reshape(slen**2, 2)
    grid_x = wcs0.xyTouv(sgrid[:, 0], sgrid[:, 1])[0]
    grid_y = wcs0.xyTouv(sgrid[:, 0], sgrid[:, 1])[1]

    grid = torch.empty(size=(0, 2))
    for dither in dithers:
        wcs1 = galsim.OffsetWCS(scale=scale, origin=galsim.PositionD(dither))
        x, y = wcs1.uvToxy(grid_x, grid_y)
        x_grid = (x / slen) * 2 + (-1)
        y_grid = (y / slen) * 2 + (-1)
        grid = torch.cat(
            [grid, torch.stack((torch.tensor(x_grid), torch.tensor(y_grid)), -1)], dim=0
        )

    interped_images = []
    input = torch.tensor(images[:]).reshape(len(dithers), 1, slen, slen).float()
    grids = grid.reshape(len(dithers), slen, slen, 2).float()
    interped_images.append(F.grid_sample(input, grids, align_corners=False))

    # reshape and crop 1 pixel on each side
    interped_images = torch.tensor(interped_images[:][0]).reshape(len(dithers), slen, slen)
    interped_images_cropped = interped_images[:, 1 : slen - 1, 1 : slen - 1]
    return interped_images_cropped
