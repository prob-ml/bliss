import matplotlib.pyplot as plt
import torch


def plot_image(
    fig,
    image,
    true_locs=None,
    estimated_locs=None,
    vmin=None,
    vmax=None,
    add_colorbar=False,
    global_fig=None,
    diverging_cmap=False,
    alpha=1,
):

    # locations are coordinates in the image, on scale from 0 to 1

    slen = image.shape[-1]

    if diverging_cmap:
        if vmax is None:
            vmax = image.abs().max()
        im = fig.matshow(image, vmin=-vmax, vmax=vmax, cmap=plt.get_cmap("bwr"))
    else:
        im = fig.matshow(image, vmin=vmin, vmax=vmax)

    if not (true_locs is None):
        assert len(true_locs.shape) == 2
        assert true_locs.shape[1] == 2
        fig.scatter(
            x=true_locs[:, 1] * (slen - 1),
            y=true_locs[:, 0] * (slen - 1),
            color="r",
            marker="x",
        )

    if not (estimated_locs is None):
        assert len(estimated_locs.shape) == 2
        assert estimated_locs.shape[1] == 2
        fig.scatter(
            x=estimated_locs[:, 1] * (slen - 1),
            y=estimated_locs[:, 0] * (slen - 1),
            color="b",
            marker="o",
            alpha=alpha,
        )

    if add_colorbar:
        assert global_fig is not None
        global_fig.colorbar(im, ax=fig)


def get_sublocs(full_locs, full_slen, sub_slen, x0, x1):
    # get locations in the subimage
    assert torch.all(full_locs <= 1)
    assert torch.all(full_locs >= 0)

    _full_locs = full_locs * (full_slen - 1)

    which_locs = (
        (_full_locs[:, 0] > x0)
        & (_full_locs[:, 0] < (x0 + sub_slen - 1))
        & (_full_locs[:, 1] > x1)
        & (_full_locs[:, 1] < (x1 + sub_slen - 1))
    )

    locs = (_full_locs[which_locs, :] - torch.Tensor([[x0, x1]])) / (sub_slen - 1)

    return locs


def plot_subimage(
    fig,
    full_image,
    full_true_locs,
    x0,
    x1,
    subimage_slen,
    full_est_locs=None,
    vmin=None,
    vmax=None,
    add_colorbar=False,
    global_fig=None,
    diverging_cmap=False,
    alpha=1,
):

    assert len(full_image.shape) == 2

    # full_est_locs and full_true_locs are locations in the coordinates of the
    # full image, in pixel units, scaled between 0 and 1
    # x0,x1 are the edges of the subimage.

    # trim image to subimage
    subimage = full_image[x0 : (x0 + subimage_slen), x1 : (x1 + subimage_slen)]

    # get slens,
    full_slen = full_image.size(-1)
    sub_slen = subimage.size(-1)

    est_locs = get_sublocs(full_est_locs, full_slen, sub_slen, x0, x1)
    true_locs = get_sublocs(full_true_locs, full_slen, sub_slen, x0, x1)

    plot_image(
        fig,
        subimage,
        true_locs=true_locs,
        estimated_locs=est_locs,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=add_colorbar,
        global_fig=global_fig,
        diverging_cmap=diverging_cmap,
        alpha=alpha,
    )
