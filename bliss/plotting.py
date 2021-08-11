from mpl_toolkits.axes_grid1 import make_axes_locatable


def _plot_locs(ax, slen, border_padding, locs, color="r", marker="x", s=20, prob_galaxy=None):
    assert len(locs.shape) == 2
    assert locs.shape[1] == 2
    assert isinstance(slen, int)
    assert isinstance(border_padding, int)
    if prob_galaxy is not None:
        assert len(prob_galaxy.shape) == 1

    x = locs[:, 1] * slen - 0.5 + border_padding
    y = locs[:, 0] * slen - 0.5 + border_padding
    for i, (xi, yi) in enumerate(zip(x, y)):
        if xi > border_padding and yi > border_padding:
            ax.scatter(xi, yi, color=color, marker=marker, s=s)
            if prob_galaxy is not None:
                ax.annotate(f"{prob_galaxy[i]:.2f}", (xi, yi), color=color, fontsize=8)


def plot_image_locs(
    ax,
    slen,
    border_padding,
    true_locs=None,
    est_locs=None,
    prob_galaxy=None,
    colors=("r", "b"),
    s=20,
    markers=("x", "+"),
    borders=True,
):
    # prob_galaxy is used to indicate confidence on prediction via marker size.

    # delineate border
    if borders:
        ax.axvline(border_padding, color="w")
        ax.axvline(border_padding + slen, color="w")
        ax.axhline(border_padding, color="w")
        ax.axhline(border_padding + slen, color="w")

    assert isinstance(border_padding, int)
    if true_locs is not None:
        _plot_locs(
            ax,
            slen,
            border_padding,
            true_locs,
            color=colors[0],
            marker=markers[0],
            s=s,
            prob_galaxy=None,
        )

    if est_locs is not None:
        s2 = 2.0 * s if markers[1] == "+" else s
        _plot_locs(
            ax,
            slen,
            border_padding,
            est_locs,
            color=colors[1],
            marker=markers[1],
            s=s2,
            prob_galaxy=prob_galaxy,
        )


def plot_image(fig, ax, image, vmin=None, vmax=None):

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(image, vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=cax, orientation="vertical")
