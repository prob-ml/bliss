import torch
from torch import nn
from . import const


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
    image_patches = None
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
    This function is used in conjunction with tile_images above. This records (x0, x1) indices
    each image patch comes from.
    :param image_xlen: The x side length of the image in pixels.
    :param image_ylen: The y side length of the image in pixels.
    :param subimage_slen: The side length of the subimage (usually the patch) in pixels.
    :param step: separation between each patch/subimage.
    :return: tile_coords, a torch.LongTensor
    """
    assert torch.cuda.is_available(), "requires use of cuda. "

    nx_patches = ((image_xlen - subimage_slen) // step) + 1
    ny_patches = ((image_ylen - subimage_slen) // step) + 1
    n_patches = nx_patches * ny_patches

    def return_coords(i):
        return [(i // ny_patches) * step,
                (i % ny_patches) * step]

    tile_coords = torch.LongTensor([return_coords(i) for i in range(n_patches)]).cuda()

    return tile_coords


def bring_to_front(n_source_params, n_sources, is_on_array, source_params, locs):
    # puts all the on sources in front
    is_on_array_full = const.get_is_on_from_n_sources(n_sources, n_sources.max())
    indx = is_on_array_full.clone()
    indx[indx == 1] = torch.nonzero(is_on_array)[:, 1]

    new_source_params = (torch.gather(source_params, dim=1, index=indx.unsqueeze(2).repeat(1, 1, n_source_params)) *
                         is_on_array_full.float().unsqueeze(2))
    new_locs = (torch.gather(locs, dim=1, index=indx.unsqueeze(2).repeat(1, 1, 2)) *
                is_on_array_full.float().unsqueeze(2))

    return new_source_params, new_locs, is_on_array_full


def get_params_in_patches(tile_coords, locs, source_params, slen, subimage_slen,
                          edge_padding=0):
    """
    Pass in the tile coordinates, the
    :param tile_coords:
    :param locs:
    :param source_params:
    :param slen:
    :param subimage_slen:
    :param edge_padding:
    :return:
    """
    # only used in running network so need cuda
    assert torch.cuda.is_available()
    # locs are the coordinates in the full image, in coordinates between 0-1
    assert torch.all(locs <= 1.)
    assert torch.all(locs >= 0.)

    n_patches = tile_coords.shape[0]  # number of patches in a full image
    fullimage_batchsize = locs.shape[0]  # number of full images

    subimage_batchsize = n_patches * fullimage_batchsize  # total number of patches

    max_sources = locs.shape[1]

    tile_coords = tile_coords.unsqueeze(0).unsqueeze(2).float()
    locs = locs * (slen - 1)
    which_locs_array = (locs.unsqueeze(1) > tile_coords + edge_padding - 0.5) & \
                       (locs.unsqueeze(1) < tile_coords - 0.5 + subimage_slen - edge_padding) & \
                       (locs.unsqueeze(1) != 0)
    which_locs_array = (which_locs_array[:, :, :, 0] * which_locs_array[:, :, :, 1]).float()

    patch_locs = \
        (which_locs_array.unsqueeze(3) * locs.unsqueeze(1) -
         (tile_coords + edge_padding - 0.5)).view(subimage_batchsize, max_sources, 2) / \
        (subimage_slen - 2 * edge_padding)
    patch_locs = torch.relu(patch_locs)  # by subtracting off, some are negative now; just set these to 0
    if source_params is not None:
        assert fullimage_batchsize == source_params.shape[0]
        assert max_sources == source_params.shape[1]
        n_source_params = source_params.shape[2]
        patch_source_params = \
            (which_locs_array.unsqueeze(3) * source_params.unsqueeze(1)).view(subimage_batchsize, max_sources,
                                                                              n_source_params)
    else:
        patch_source_params = torch.zeros(patch_locs.shape[0], patch_locs.shape[1], 1)
        n_source_params = 1

    # sort locs so all the zeros are at the end
    is_on_array = which_locs_array.view(subimage_batchsize, max_sources).type(torch.bool).cuda()
    n_sources_per_patch = is_on_array.float().sum(dim=1).type(torch.LongTensor).cuda()

    patch_source_params, patch_locs, patch_is_on_array = bring_to_front(n_source_params, n_sources_per_patch,
                                                                        is_on_array,
                                                                        patch_source_params, patch_locs)

    return patch_locs, patch_source_params, n_sources_per_patch, patch_is_on_array


def get_full_params_from_patch_params(patch_locs, patch_source_params,
                                      tile_coords,
                                      full_slen,
                                      stamp_slen,
                                      edge_padding):
    # NOTE: off sources should have patch_locs == 0 and patch_source_params == 0

    # reshaped before passing in into shape (batchsize * n_image_patches, -1, self.n_source_params)
    assert (patch_source_params.shape[0] % tile_coords.shape[0]) == 0
    batchsize = int(patch_source_params.shape[0] / tile_coords.shape[0])

    assert (patch_source_params.shape[0] % batchsize) == 0
    n_sources_in_batch = int(patch_source_params.shape[0] * patch_source_params.shape[1] / batchsize)

    latent_dim = patch_source_params.shape[2]  # = n_bands in the case of fluxes.
    source_params = patch_source_params.view(batchsize, n_sources_in_batch, latent_dim)

    scale = (stamp_slen - 2 * edge_padding)
    bias = tile_coords.repeat(batchsize, 1).unsqueeze(1).float() + edge_padding - 0.5
    locs = (patch_locs * scale + bias) / (full_slen - 1)

    locs = locs.view(batchsize, n_sources_in_batch, 2)

    patch_is_on_bool = (source_params > 0).any(2).float()  # if source_param in any latent_dim is nonzero
    n_sources = torch.sum(patch_is_on_bool > 0, dim=1)

    # puts all the on stars in front (of each patch subarray)
    is_on_array_full = const.get_is_on_from_n_sources(n_sources, n_sources.max())
    indx = is_on_array_full.clone()
    indx[indx == 1] = torch.nonzero(patch_is_on_bool)[:, 1]

    source_params, locs, _ = bring_to_front(latent_dim, n_sources, patch_is_on_bool, source_params, locs)
    return locs, source_params, n_sources


def trim_images(images, edge_padding):
    slen = images.shape[-1] - edge_padding

    return images[:, :, edge_padding:slen, edge_padding:slen]
