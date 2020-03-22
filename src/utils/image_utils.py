import torch
from torch import nn
from GalaxyModel.src import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _extract_patches_2d(img, patch_shape, step=None, batch_first=False):
    """
    Take in an image (tensor) and the shape of the patch we want to separate it into and
    return the patches also as a tensor.

    Taken from: https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78

    :param img:
    :type img: class: `torch.Tensor`
    :param patch_shape:
    :param step:
    :param batch_first:
    :return: A tensor of patches.
    :rtype: class: `torch.Tensor`
    """
    if step is None:
        step = [1.0, 1.0]

    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if img.size(2) < patch_H:
        num_padded_H_Top = (patch_H - img.size(2)) // 2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if img.size(3) < patch_W:
        num_padded_W_Left = (patch_W - img.size(3)) // 2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0]) if (isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W * step[1]) if (isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if (img.size(2) - patch_H) % step_int[0] != 0:
        patches_fold_H = torch.cat((patches_fold_H, img[:, :, -patch_H:, ].permute(0, 1, 3, 2).unsqueeze(2)), dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if (img.size(3) - patch_W) % step_int[1] != 0:
        patches_fold_HW = torch.cat(
            (patches_fold_HW, patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)
    patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)
    if batch_first:
        patches = patches.permute(1, 0, 2, 3, 4)
    return patches


def tile_images(images, subimage_slen, step):
    """
    Breaks up a large image into smaller patches. Each patch has size subimage_slen x subimage_slen, where
    patches per image  is (slen - subimage_sel / step)**2.

    NOTE: input and output are torch tensors, not numpy arrays.

    :param images: A tensor of size (batchsize x n_bands x slen x slen)
    :type images: class:`torch.Tensor`
    :param subimage_slen: The side length of each patch.
    :param step:
    :return: image_patches, output tensor of shape:
             (batchsize * patches per image) x n_bands x subimage_slen x subimage_slen
    :rtype: class:`torch.Tensor`
    """

    assert len(images.shape) == 4

    image_xlen = images.shape[2]
    image_ylen = images.shape[3]

    # My tile coords doesn't work otherwise ...
    assert (image_xlen - subimage_slen) % step == 0
    assert (image_ylen - subimage_slen) % step == 0

    n_bands = images.shape[1]
    for b in range(n_bands):
        image_patches_b = _extract_patches_2d(images[:, b:(b + 1), :, :],
                                              patch_shape=[subimage_slen, subimage_slen],
                                              step=[step, step],
                                              batch_first=True).reshape(-1, 1, subimage_slen, subimage_slen)

        if b == 0:
            image_patches = image_patches_b
        else:
            image_patches = torch.cat((image_patches, image_patches_b), dim=1)

    return image_patches


def get_tile_coords(image_xlen, image_ylen, subimage_slen, step):
    """
    This function is used in conjuction with tile_images above. This records (x0, x1) indices
    each image patch comes from.
    :param image_xlen: The x side length of the image in pixels.
    :param image_ylen: The y side length of the image in pixels.
    :param subimage_slen: The side length of the subimage (usually the patch) in pixels.
    :param step: separation between each patch/subimage.
    :return: tile_coords, a torch.LongTensor
    """

    nx_patches = ((image_xlen - subimage_slen) // step) + 1
    ny_patches = ((image_ylen - subimage_slen) // step) + 1
    n_patches = nx_patches * ny_patches

    def return_coords(i):
        return [(i // ny_patches) * step,
                (i % ny_patches) * step]

    tile_coords = torch.LongTensor([return_coords(i) for i in range(n_patches)]).to(device)

    return tile_coords


def get_params_in_patches(tile_coords, locs, fluxes, slen, subimage_slen,
                          edge_padding=0):
    """
    Pass in the tile coordinates, the
    :param tile_coords:
    :param locs:
    :param fluxes:
    :param slen:
    :param subimage_slen:
    :param edge_padding:
    :return:
    """
    # locs are the coordinates in the full image, in coordinates between 0-1
    assert torch.all(locs <= 1.)
    assert torch.all(locs >= 0.)

    n_patches = tile_coords.shape[0]  # number of patches in a full image
    fullimage_batchsize = locs.shape[0]  # number of full images

    subimage_batchsize = n_patches * fullimage_batchsize  # total number of patches

    max_stars = locs.shape[1]

    tile_coords = tile_coords.unsqueeze(0).unsqueeze(2).float()
    locs = locs * (slen - 1)
    which_locs_array = (locs.unsqueeze(1) > tile_coords + edge_padding - 0.5) & \
                       (locs.unsqueeze(1) < tile_coords - 0.5 + subimage_slen - edge_padding) & \
                       (locs.unsqueeze(1) != 0)
    which_locs_array = (which_locs_array[:, :, :, 0] * which_locs_array[:, :, :, 1]).float()

    patch_locs = \
        (which_locs_array.unsqueeze(3) * locs.unsqueeze(1) -
         (tile_coords + edge_padding - 0.5)).view(subimage_batchsize, max_stars, 2) / \
        (subimage_slen - 2 * edge_padding)
    patch_locs = torch.relu(patch_locs)  # by subtracting off, some are negative now; just set these to 0
    if fluxes is not None:
        assert fullimage_batchsize == fluxes.shape[0]
        assert max_stars == fluxes.shape[1]
        n_bands = fluxes.shape[2]
        patch_fluxes = \
            (which_locs_array.unsqueeze(3) * fluxes.unsqueeze(1)).view(subimage_batchsize, max_stars, n_bands)
    else:
        patch_fluxes = torch.zeros(patch_locs.shape[0], patch_locs.shape[1], 1)
        n_bands = 1

    # sort locs so all the zeros are at the end
    is_on_array = which_locs_array.view(subimage_batchsize, max_stars).type(torch.bool).to(device)
    n_stars_per_patch = is_on_array.float().sum(dim=1).type(torch.LongTensor).to(device)

    is_on_array_sorted = utils.get_is_on_from_n_stars(n_stars_per_patch, n_stars_per_patch.max())

    indx = is_on_array_sorted.clone()
    indx[indx == 1] = torch.nonzero(is_on_array)[:, 1]

    patch_fluxes = torch.gather(patch_fluxes, dim=1, index=indx.unsqueeze(2).repeat(1, 1, n_bands)) * \
                   is_on_array_sorted.float().unsqueeze(2)
    patch_locs = torch.gather(patch_locs, dim=1, index=indx.unsqueeze(2).repeat(1, 1, 2)) * \
                 is_on_array_sorted.float().unsqueeze(2)

    patch_is_on_array = is_on_array_sorted

    return patch_locs, patch_fluxes, n_stars_per_patch, patch_is_on_array


def get_full_params_from_patch_params(patch_locs, patch_fluxes,
                                      tile_coords,
                                      full_slen,
                                      stamp_slen,
                                      edge_padding):
    # off stars should have patch_locs == 0 and patch_fluxes == 0

    assert (patch_fluxes.shape[0] % tile_coords.shape[0]) == 0
    batchsize = int(patch_fluxes.shape[0] / tile_coords.shape[0])

    assert (patch_fluxes.shape[0] % batchsize) == 0
    n_stars_in_batch = int(patch_fluxes.shape[0] * patch_fluxes.shape[1] / batchsize)

    n_bands = patch_fluxes.shape[2]
    fluxes = patch_fluxes.view(batchsize, n_stars_in_batch, n_bands)

    scale = (stamp_slen - 2 * edge_padding)
    bias = tile_coords.repeat(batchsize, 1).unsqueeze(1).float() + edge_padding - 0.5
    locs = (patch_locs * scale + bias) / (full_slen - 1)

    locs = locs.view(batchsize, n_stars_in_batch, 2)

    patch_is_on_bool = (fluxes > 0).any(2).float()  # if flux in any band is nonzero
    n_stars = torch.sum(patch_is_on_bool > 0, dim=1)

    # puts all the on stars in front
    is_on_array_full = utils.get_is_on_from_n_stars(n_stars, n_stars.max())
    indx = is_on_array_full.clone()
    indx[indx == 1] = torch.nonzero(patch_is_on_bool)[:, 1]

    fluxes = torch.gather(fluxes, dim=1, index=indx.unsqueeze(2).repeat(1, 1, n_bands)) * \
             is_on_array_full.float().unsqueeze(2)
    locs = torch.gather(locs, dim=1, index=indx.unsqueeze(2).repeat(1, 1, 2)) * \
           is_on_array_full.float().unsqueeze(2)

    return locs, fluxes, n_stars


def trim_images(images, edge_padding):
    slen = images.shape[-1] - edge_padding

    return images[:, :, edge_padding:slen, edge_padding:slen]
