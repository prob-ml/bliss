from matplotlib import pyplot as plt


def subtract(x, y):
    return x - y


def plot_residuals(image, recon_mean, resid_fun=subtract):

    assert len(image.shape) == 2
    assert image.shape == recon_mean.shape

    fig, ax = plt.subplots(1, 3, figsize=(15, 3))

    im0 = ax[0].matshow(image.cpu())
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].matshow(recon_mean.detach().cpu())
    fig.colorbar(im1, ax=ax[1])

    resid = resid_fun(image, recon_mean).detach().cpu()
    vmax = resid.abs().max()
    im2 = ax[2].matshow(resid, vmax=vmax, vmin=-vmax, cmap=plt.get_cmap("bwr"))
    fig.colorbar(im2, ax=ax[2])

    return fig, ax


def scatter_plot(locs, ax, color="r", marker="x", label=None):
    ax.scatter(locs[:, 0], locs[:, 1], color=color, marker=marker, label=label)


def plot_locations(locs, is_gal, n, ax):
    is_gal = is_gal[0:n]
    locs = locs[0:n]

    scatter_plot(locs[is_gal == 1], ax, marker="x", color="red", label="galaxy")

    scatter_plot(locs[is_gal == 0], ax, marker="+", color="blue", label="star")
