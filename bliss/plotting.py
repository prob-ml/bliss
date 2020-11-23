from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_locs(ax, slen, border_padding, locs, color="r", marker="x", s=1):
    assert len(locs.shape) == 2
    assert locs.shape[1] == 2
    assert isinstance(slen, int)
    assert isinstance(border_padding, int)
    ax.scatter(
        x=locs[:, 1] * (slen - 1) - 0.5 + border_padding,
        y=locs[:, 0] * (slen - 1) - 0.5 + border_padding,
        color=color,
        marker=marker,
        s=s,
    )


def plot_image_locs(
    ax,
    slen,
    border_padding,
    true_locs=None,
    est_locs=None,
    colors=("r", "b"),
    s=20,
):

    assert isinstance(border_padding, int)
    if true_locs is not None:
        plot_locs(ax, slen, border_padding, true_locs, color=colors[0], marker="x", s=s)

    if est_locs is not None:
        s2 = 2 * s
        plot_locs(ax, slen, border_padding, est_locs, color=colors[1], marker="+", s=s2)


def plot_image(
    fig,
    ax,
    image,
    vmin=None,
    vmax=None,
):

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(image, vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=cax, orientation="vertical")
