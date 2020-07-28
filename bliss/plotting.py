import matplotlib.pyplot as plt
import torch


def plot_image(
    fig,
    image,
    true_locs=None,
    estimated_locs=None,
    vmin=None,
    vmax=None,
    add_colorbar=True,
    global_fig=None,
    alpha=1,
    marker_size=1,
):

    # locations are coordinates in the image, on scale from 0 to 1

    slen = image.shape[-1]

    im = fig.matshow(image, vmin=vmin, vmax=vmax)

    if not (true_locs is None):
        assert len(true_locs.shape) == 2
        assert true_locs.shape[1] == 2
        fig.scatter(
            x=true_locs[:, 1] * (slen - 1),
            y=true_locs[:, 0] * (slen - 1),
            color="r",
            marker="x",
            s=marker_size,
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
            s=marker_size,
        )

    if add_colorbar:
        assert global_fig is not None
        global_fig.colorbar(im, ax=fig)
