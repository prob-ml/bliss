from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor


def plot_image(fig, ax, image, vrange=None):
    vmin = image.min().item() if vrange is None else vrange[0]
    vmax = image.max().item() if vrange is None else vrange[1]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.matshow(image, vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=cax, orientation="vertical")


def plot_locs(ax, slen, bpad, locs, color="r", marker="x", s=20, prob_galaxy=None):
    assert len(locs.shape) == 2
    assert locs.shape[1] == 2
    assert isinstance(slen, int)
    assert isinstance(bpad, int)
    if prob_galaxy is not None:
        assert len(prob_galaxy.shape) == 1

    x = locs[:, 1] * slen - 0.5 + bpad
    y = locs[:, 0] * slen - 0.5 + bpad
    for i, (xi, yi) in enumerate(zip(x, y)):
        if xi > bpad and yi > bpad:
            ax.scatter(xi, yi, color=color, marker=marker, s=s)
            if prob_galaxy is not None:
                ax.annotate(f"{prob_galaxy[i]:.2f}", (xi, yi), color=color, fontsize=8)


def plot_image_and_locs(
    idx: int,
    fig: Figure,
    ax: Axes,
    images,
    slen: int,
    true_params: dict,
    estimate: dict = None,
    labels: list = None,
    annotate_axis: bool = False,
    add_borders: bool = False,
    vrange: tuple = None,
    prob_galaxy: Tensor = None,
):
    # collect all necessary parameters to plot
    assert images.shape[1] == 1, "Only 1 band supported."
    bpad = int((images.shape[-1] - slen) / 2)

    image = images[idx, 0].cpu().numpy()

    # true parameters on full image.
    true_n_sources = true_params["n_sources"][idx].cpu().numpy()
    true_locs = true_params["locs"][idx].cpu().numpy()
    true_galaxy_bool = true_params["galaxy_bool"][idx].cpu().numpy()
    true_star_bool = true_params["star_bool"][idx].cpu().numpy()
    true_galaxy_locs = true_locs * true_galaxy_bool
    true_star_locs = true_locs * true_star_bool

    # convert tile estimates to full parameterization for plotting
    if estimate is not None:
        n_sources = estimate["n_sources"][idx].cpu().numpy()
        locs = estimate["locs"][idx].cpu().numpy()
        galaxy_bool = estimate["galaxy_bool"][idx].cpu().numpy()
        star_bool = estimate["star_bool"][idx].cpu().numpy()
        galaxy_locs = locs * galaxy_bool
        star_locs = locs * star_bool

    if prob_galaxy is not None:
        prob_galaxy = prob_galaxy[idx].cpu().numpy().reshape(-1)

    # annotate useful information around the axis
    if annotate_axis and estimate is not None:
        ax.set_xlabel(f"True num: {true_n_sources.item()}; Est num: {n_sources.item()}")

    # (optionally) add white border showing where centers of stars and galaxies can be
    if add_borders:
        ax.axvline(bpad, color="w")
        ax.axvline(bpad + slen, color="w")
        ax.axhline(bpad, color="w")
        ax.axhline(bpad + slen, color="w")

    # plot image first
    vmin = image.min().item() if vrange is None else vrange[0]
    vmax = image.max().item() if vrange is None else vrange[1]
    plot_image(fig, ax, image, vrange=(vmin, vmax))

    # plot locations
    plot_locs(ax, slen, bpad, true_galaxy_locs, "r", "x", s=20, prob_galaxy=None)
    plot_locs(ax, slen, bpad, true_star_locs, "c", "x", s=20, prob_galaxy=None)

    if estimate is not None:
        plot_locs(ax, slen, bpad, galaxy_locs, "b", "+", s=30, prob_galaxy=prob_galaxy)
        plot_locs(ax, slen, bpad, star_locs, "m", "+", s=30, prob_galaxy=prob_galaxy)

    if labels is not None:
        assert len(labels) == 4
        colors = ["r", "b", "c", "m"]
        markers = ["x", "+", "x", "+"]
        sizes = [25, 35, 25, 35]
        for l, c, m, s in zip(labels, colors, markers, sizes):
            if l is not None:
                ax.scatter(0, 0, color=c, s=s, marker=m, label=l)
        ax.legend(
            bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
        )
