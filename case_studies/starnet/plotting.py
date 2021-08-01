import torch
from matplotlib import pyplot as plt

from bliss.models.encoder import get_is_on_from_n_sources


def plot_image(axarr, image, x0=0, x1=0, slen0=100, slen1=100):

    subimage = image[x0 : (x0 + slen0), x1 : (x1 + slen1)]
    vmin = subimage.min()
    vmax = subimage.max()

    im = axarr.matshow(image.cpu(), cmap=plt.cm.get_cmap("gray"), vmin=vmin, vmax=vmax)
    axarr.set_ylim(x0 + slen0, x0)
    axarr.set_xlim(x1, x1 + slen1)

    return im


def plot_locations(locs, slen, border_padding, ax, marker="o", color="b"):
    ax.scatter(
        locs[:, 1].cpu() * slen - 0.5 + border_padding,
        locs[:, 0].cpu() * slen - 0.5 + border_padding,
        marker=marker,
        color=color,
    )


def get_params_in_tiles(tile_coords, locs, fluxes, slen, subimage_slen, edge_padding=0):

    # locs are the coordinates in the full image, in coordinates between 0-1
    assert torch.all(locs <= 1.0)
    assert torch.all(locs >= 0.0)

    n_tiles = tile_coords.shape[0]  # number of tiles in a full image
    fullimage_batchsize = locs.shape[0]  # number of full images

    subimage_batchsize = n_tiles * fullimage_batchsize  # total number of tiles

    max_stars = locs.shape[1]

    tile_coords = tile_coords.unsqueeze(0).unsqueeze(2).float()
    locs = locs * (slen - 1)
    which_locs_array = (
        (locs.unsqueeze(1) > tile_coords + edge_padding - 0.5)
        & (locs.unsqueeze(1) < tile_coords - 0.5 + subimage_slen - edge_padding)
        & (locs.unsqueeze(1) != 0)
    )
    which_locs_array = (which_locs_array[:, :, :, 0] * which_locs_array[:, :, :, 1]).float()

    tile_locs = (
        which_locs_array.unsqueeze(3) * locs.unsqueeze(1) - (tile_coords + edge_padding - 0.5)
    ).view(subimage_batchsize, max_stars, 2) / (subimage_slen - 2 * edge_padding)
    tile_locs = torch.relu(
        tile_locs
    )  # by subtracting off, some are negative now; just set these to 0
    if fluxes is not None:
        assert fullimage_batchsize == fluxes.shape[0]
        assert max_stars == fluxes.shape[1]
        n_bands = fluxes.shape[2]
        tile_fluxes = (which_locs_array.unsqueeze(3) * fluxes.unsqueeze(1)).view(
            subimage_batchsize, max_stars, n_bands
        )
    else:
        tile_fluxes = torch.zeros(tile_locs.shape[0], tile_locs.shape[1], 1)
        n_bands = 1

    # sort locs so all the zeros are at the end
    is_on_array = which_locs_array.view(subimage_batchsize, max_stars).type(torch.bool)
    n_stars_per_tile = is_on_array.float().sum(dim=1).type(torch.LongTensor)

    is_on_array_sorted = get_is_on_from_n_sources(n_stars_per_tile, n_stars_per_tile.max())

    indx = is_on_array_sorted.clone().long()
    indx[indx == 1] = torch.nonzero(is_on_array, as_tuple=False)[:, 1]

    tile_fluxes = torch.gather(
        tile_fluxes, dim=1, index=indx.unsqueeze(2).repeat(1, 1, n_bands)
    ) * is_on_array_sorted.float().unsqueeze(2)
    tile_locs = torch.gather(
        tile_locs, dim=1, index=indx.unsqueeze(2).repeat(1, 1, 2)
    ) * is_on_array_sorted.float().unsqueeze(2)

    tile_is_on_array = is_on_array_sorted

    return tile_locs, tile_fluxes, n_stars_per_tile, tile_is_on_array
