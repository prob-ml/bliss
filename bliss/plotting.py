"""Common functions to plot results."""
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_detections(images, true_cat, est_cat, nrows, img_ids, margin_px):
    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(20, 20))
    axes = axes.flatten() if nrows > 1 else [axes]  # flatten

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(img_ids):  # don't plot on this ax if there aren't enough images
            break

        img_id = img_ids[ax_idx]
        true_n_sources = true_cat.n_sources[img_id].item()
        n_sources = est_cat.n_sources[img_id].item()
        ax.set_xlabel(f"True num: {true_n_sources}; Est num: {n_sources}")

        # add white border showing where centers of stars and galaxies can be
        ax.axvline(margin_px, color="w")
        ax.axvline(images.shape[-1] - margin_px, color="w")
        ax.axhline(margin_px, color="w")
        ax.axhline(images.shape[-2] - margin_px, color="w")

        # plot image first
        image = images[img_id, 0].cpu().numpy()
        vmin = image.min().item()
        vmax = image.max().item()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.matshow(image, vmin=vmin, vmax=vmax, cmap="viridis")
        fig.colorbar(im, cax=cax, orientation="vertical")

        true_cat.plot_plocs(ax, img_id, "galaxy", bp=margin_px, color="r", marker="x", s=20)
        true_cat.plot_plocs(ax, img_id, "star", bp=margin_px, color="m", marker="x", s=20)
        est_cat.plot_plocs(ax, img_id, "all", bp=margin_px, color="b", marker="+", s=30)

        if ax_idx == 0:
            ax.scatter(None, None, color="r", marker="x", s=20, label="t.gal")
            ax.scatter(None, None, color="m", marker="x", s=20, label="t.star")
            ax.scatter(None, None, color="b", marker="+", s=30, label="p.source")
            ax.legend(
                bbox_to_anchor=(0.0, 1.2, 1.0, 0.102),
                loc="lower left",
                ncol=2,
                mode="expand",
                borderaxespad=0.0,
            )

    fig.tight_layout()
    return fig
