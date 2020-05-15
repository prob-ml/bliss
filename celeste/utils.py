import json
from pathlib import Path
from os.path import dirname

import numpy as np
import torch
from torch.distributions import categorical, Normal
from torch import nn

# global paths
root_path = Path(dirname(dirname(__file__)))

data_path = root_path.joinpath("data")
config_path = root_path.joinpath("config")
reports_path = root_path.joinpath("reports")
results_path = root_path.joinpath("results")

# global variables
image_h5_name = "images"
background_h5_name = "background"

# make codebase device agnostic, but also create all tensors directly in the gpu when possible.
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")
if use_cuda:
    default_device = 0
    torch.cuda.set_device(default_device)
    device = torch.device(default_device)

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


# let the user change device defined in this module.
def set_device(device_id=None, no_cuda=False):

    if not no_cuda:
        torch.cuda.set_device(device_id)

    global device
    device = torch.device(device_id) if not no_cuda else torch.device("cpu")


def load_data_params_from_args(params_file_name, args):
    params_path = config_path.joinpath(params_file_name)
    with open(params_path, "r") as fp:
        data_params = json.load(fp)

    args_dict = vars(args)
    for k in data_params:
        if k in args_dict and args_dict[k] is not None:
            data_params[k] = args_dict[k]
    return data_params


#############################
# Tensor functions
############################


def get_is_on_from_n_sources(n_sources, max_sources):
    """
    Return a boolean array of shape=(batchsize, max_sources) whose (k,l)th entry indicates
    whether there are more than l sources on the kth batch.
    :param n_sources:
    :param max_sources:
    :return:
    """
    assert len(n_sources.shape) == 1

    batchsize = len(n_sources)
    is_on_array = LongTensor(batchsize, max_sources).zero_()

    for i in range(max_sources):
        is_on_array[:, i] = n_sources > i

    return is_on_array


def get_is_on_from_tile_n_sources_2d(tile_n_sources, max_sources):
    """

    :param tile_n_sources: A tensor of shape (n_samples x n_tiles), indicating the number of sources
                            at sample i, batch j. (n_samples = batchsize)
    :type tile_n_sources: class: `torch.Tensor`
    :param max_sources:
    :type max_sources: int
    :return:
    """
    assert not torch.any(torch.isnan(tile_n_sources))
    assert torch.all(tile_n_sources >= 0)
    assert torch.all(tile_n_sources <= max_sources)

    n_samples = tile_n_sources.shape[0]
    batchsize = tile_n_sources.shape[1]

    is_on_array = LongTensor(n_samples, batchsize, max_sources).zero_()

    for i in range(max_sources):
        is_on_array[:, :, i] = tile_n_sources > i

    return is_on_array


def get_one_hot_encoding_from_int(z, n_classes):
    z = z.long()

    assert len(torch.unique(z)) <= n_classes

    z_one_hot = FloatTensor(len(z), n_classes).zero_()
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot


#############################
# Sampling functions
############################


def draw_pareto(f_min, alpha, shape):
    uniform_samples = FloatTensor(*shape).uniform_()
    return f_min / (1.0 - uniform_samples) ** (1 / alpha)


def draw_pareto_maxed(f_min, f_max, alpha, shape):
    # draw pareto conditioned on being less than f_max

    pareto_samples = draw_pareto(f_min, alpha, shape)

    while torch.any(pareto_samples > f_max):
        indx = pareto_samples > f_max
        pareto_samples[indx] = draw_pareto(f_min, alpha, [torch.sum(indx).item()])

    return pareto_samples


def sample_class_weights(class_weights, n_samples=1):
    """
    Draw a sample from Categorical variable with
    probabilities class_weights.
    """

    assert not torch.any(torch.isnan(class_weights))
    cat_rv = categorical.Categorical(probs=class_weights)
    return cat_rv.sample((n_samples,)).detach().squeeze()


# eval_normal_logprob
#
# def eval_normal_logprob(x, mu, logvar):
#     return (
#         -0.5 * logvar
#         - 0.5 * (x - mu) ** 2 / (torch.exp(logvar) + 1e-5)
#         - 0.5 * np.log(2 * np.pi)
#     )


#############################
# Image utility
############################


def _extract_ptiles_2d(img, tile_shape, step=None, batch_first=False):
    """
    Take in an image (tensor) and the shape of the padded tile
    we want to separate it into and
    return the padded tiles also as a tensor.

    Taken from: https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78

    :param img:
    :type img: class: `torch.Tensor`
    :param tile_shape:
    :param step:
    :param batch_first:
    :return: A tensor of padded tiles.
    :rtype: class: `torch.Tensor`
    """
    if step is None:
        step = [1.0, 1.0]

    tile_H, tile_W = tile_shape[0], tile_shape[1]
    if img.size(2) < tile_H:
        num_padded_H_Top = (tile_H - img.size(2)) // 2
        num_padded_H_Bottom = tile_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if img.size(3) < tile_W:
        num_padded_W_Left = (tile_W - img.size(3)) // 2
        num_padded_W_Right = tile_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)
    step_int = [0, 0]
    step_int[0] = int(tile_H * step[0]) if (isinstance(step[0], float)) else step[0]
    step_int[1] = int(tile_W * step[1]) if (isinstance(step[1], float)) else step[1]
    ptiles_fold_H = img.unfold(2, tile_H, step_int[0])
    if (img.size(2) - tile_H) % step_int[0] != 0:
        ptiles_fold_H = torch.cat(
            (ptiles_fold_H, img[:, :, -tile_H:,].permute(0, 1, 3, 2).unsqueeze(2)),
            dim=2,
        )
    ptiles_fold_HW = ptiles_fold_H.unfold(3, tile_W, step_int[1])
    if (img.size(3) - tile_W) % step_int[1] != 0:
        ptiles_fold_HW = torch.cat(
            (
                ptiles_fold_HW,
                ptiles_fold_H[:, :, :, -tile_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3),
            ),
            dim=3,
        )
    ptiles = ptiles_fold_HW.permute(2, 3, 0, 1, 4, 5)
    ptiles = ptiles.reshape(-1, img.size(0), img.size(1), tile_H, tile_W)
    if batch_first:
        ptiles = ptiles.permute(1, 0, 2, 3, 4)
    return ptiles


def tile_images(images, ptile_slen, step):
    """
    Breaks up a large image into smaller padded tiles.
    Each tile has size ptile_slen x ptile_slen, where
    the number of padded tiles per image  is (slen - ptile_slen / step)**2.

    NOTE: input and output are torch tensors, not numpy arrays.

    :param images: A tensor of size (batchsize x n_bands x slen x slen)
    :type images: class:`torch.Tensor`
    :param ptile_slen: The side length of each padded tile.
    :param step:
    :return: image_ptiles, output tensor of shape:
             (batchsize * ptiles per image) x n_bands x ptile_slen x ptile_slen
    :rtype: class:`torch.Tensor`
    """

    assert len(images.shape) == 4

    image_xlen = images.shape[2]
    image_ylen = images.shape[3]

    # My tile coords doesn't work otherwise ...
    assert (image_xlen - ptile_slen) % step == 0
    assert (image_ylen - ptile_slen) % step == 0

    n_bands = images.shape[1]
    image_ptiles = None
    for b in range(n_bands):
        image_ptiles_b = _extract_ptiles_2d(
            images[:, b : (b + 1), :, :],
            tile_shape=[ptile_slen, ptile_slen],
            step=[step, step],
            batch_first=True,
        ).reshape(-1, 1, ptile_slen, ptile_slen)

        if b == 0:
            image_ptiles = image_ptiles_b
        else:
            image_ptiles = torch.cat((image_ptiles, image_ptiles_b), dim=1)

    return image_ptiles


def get_ptile_coords(image_xlen, image_ylen, ptile_slen, step):
    """
    This function is used in conjunction with tile_images above. This records (x0, x1) indices
    each image padded tile comes from.
    :param image_xlen: The x side length of the image in pixels.
    :param image_ylen: The y side length of the image in pixels.
    :param ptile_slen: The side length of the padded tile in pixels.
    :param step: pixels by which to shift every padded tile.
    :return: tile_coords, a torch.LongTensor
    """

    nx_ptiles = ((image_xlen - ptile_slen) // step) + 1
    ny_ptiles = ((image_ylen - ptile_slen) // step) + 1
    n_ptiles = nx_ptiles * ny_ptiles

    def return_coords(i):
        return [(i // ny_ptiles) * step, (i % ny_ptiles) * step]

    tile_coords = torch.LongTensor([return_coords(i) for i in range(n_ptiles)]).to(
        device
    )

    return tile_coords


def bring_to_front(n_source_params, n_sources, is_on_array, source_params, locs):
    # puts all the on sources in front
    is_on_array_full = get_is_on_from_n_sources(n_sources, n_sources.max())
    indx = is_on_array_full.clone()
    indx[indx == 1] = torch.nonzero(is_on_array)[:, 1]

    new_source_params = torch.gather(
        source_params, dim=1, index=indx.unsqueeze(2).repeat(1, 1, n_source_params)
    ) * is_on_array_full.float().unsqueeze(2)
    new_locs = torch.gather(
        locs, dim=1, index=indx.unsqueeze(2).repeat(1, 1, 2)
    ) * is_on_array_full.float().unsqueeze(2)

    return new_source_params, new_locs, is_on_array_full


def get_params_in_tiles(
    tile_coords, locs, source_params, slen, ptile_slen, edge_padding=0
):
    """
    Pass in the tile coordinates, the
    :param tile_coords:
    :param locs:
    :param source_params:
    :param slen:
    :param ptile_slen:
    :param edge_padding:
    :return:
    """
    # locs are the coordinates in the full image, in coordinates between 0-1
    assert torch.all(locs <= 1.0)
    assert torch.all(locs >= 0.0)

    n_ptiles = tile_coords.shape[0]  # number of ptiles in a full image
    fullimage_batchsize = locs.shape[0]  # number of full images

    subimage_batchsize = n_ptiles * fullimage_batchsize  # total number of ptiles

    max_sources = locs.shape[1]

    tile_coords = tile_coords.unsqueeze(0).unsqueeze(2).float()
    locs = locs * (slen - 1)
    which_locs_array = (
        (locs.unsqueeze(1) > tile_coords + edge_padding - 0.5)
        & (locs.unsqueeze(1) < tile_coords - 0.5 + ptile_slen - edge_padding)
        & (locs.unsqueeze(1) != 0)
    )
    which_locs_array = (
        which_locs_array[:, :, :, 0] * which_locs_array[:, :, :, 1]
    ).float()

    tile_locs = (
        which_locs_array.unsqueeze(3) * locs.unsqueeze(1)
        - (tile_coords + edge_padding - 0.5)
    ).view(subimage_batchsize, max_sources, 2) / (ptile_slen - 2 * edge_padding)
    tile_locs = torch.relu(
        tile_locs
    )  # by subtracting off, some are negative now; just set these to 0
    if source_params is not None:
        assert fullimage_batchsize == source_params.shape[0]
        assert max_sources == source_params.shape[1]
        n_source_params = source_params.shape[2]
        tile_source_params = (
            which_locs_array.unsqueeze(3) * source_params.unsqueeze(1)
        ).view(subimage_batchsize, max_sources, n_source_params)
    else:
        tile_source_params = torch.zeros(tile_locs.shape[0], tile_locs.shape[1], 1)
        n_source_params = 1

    # sort locs so all the zeros are at the end
    is_on_array = (
        which_locs_array.view(subimage_batchsize, max_sources)
        .type(torch.bool)
        .to(device)
    )
    n_sources_per_tile = (
        is_on_array.float().sum(dim=1).type(torch.LongTensor).to(device)
    )

    tile_source_params, tile_locs, tile_is_on_array = bring_to_front(
        n_source_params, n_sources_per_tile, is_on_array, tile_source_params, tile_locs,
    )

    return tile_locs, tile_source_params, n_sources_per_tile, tile_is_on_array


def get_full_params_from_tile_params(
    tile_locs, tile_source_params, tile_coords, full_slen, stamp_slen, edge_padding
):
    # NOTE: off sources should have tile_locs == 0 and tile_source_params == 0

    # reshaped before passing in into shape (batchsize * n_image_ptiles, -1, self.n_source_params)
    assert (tile_source_params.shape[0] % tile_coords.shape[0]) == 0
    batchsize = int(tile_source_params.shape[0] / tile_coords.shape[0])

    assert (tile_source_params.shape[0] % batchsize) == 0
    n_sources_in_batch = int(
        tile_source_params.shape[0] * tile_source_params.shape[1] / batchsize
    )

    n_source_params = tile_source_params.shape[2]  # = n_bands in the case of fluxes.
    source_params = tile_source_params.view(
        batchsize, n_sources_in_batch, n_source_params
    )

    scale = stamp_slen - 2 * edge_padding
    bias = tile_coords.repeat(batchsize, 1).unsqueeze(1).float() + edge_padding - 0.5
    locs = (tile_locs * scale + bias) / (full_slen - 1)

    locs = locs.view(batchsize, n_sources_in_batch, 2)

    tile_is_on_bool = (
        (source_params > 0).any(2).float()
    )  # if source_param in any n_source_params is nonzero
    n_sources = torch.sum(tile_is_on_bool > 0, dim=1)

    # puts all the on sources in front (of each tile subarray)
    is_on_array_full = get_is_on_from_n_sources(n_sources, n_sources.max())
    indx = is_on_array_full.clone()
    indx[indx == 1] = torch.nonzero(tile_is_on_bool)[:, 1]

    source_params, locs, _ = bring_to_front(
        n_source_params, n_sources, tile_is_on_bool, source_params, locs
    )
    return locs, source_params, n_sources


def trim_images(images, edge_padding):
    slen = images.shape[-1] - edge_padding

    return images[:, :, edge_padding:slen, edge_padding:slen]
