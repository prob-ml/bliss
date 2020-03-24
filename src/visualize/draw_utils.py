import matplotlib.pyplot as plt
from utils.const import const


def draw_multiband(image, filename, figsize=(18, 18), ending='.jpg'):
    """
    Draw a single multi-band image across six bands saving into filename saving it into reports/figures/filename.pdf
    :param image:
    :param filename:
    :param figsize:
    :param ending:
    :return:
    """
    file_path = const.reports_path.joinpath(f"figures/{filename}{ending}")

    num_bands = image.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axs = axes.flatten()

    for i, ax in enumerate(axs):
        im = ax.imshow(image[i])
        fig.colorbar(im, ax=ax, orientation='vertical')

    fig.savefig(file_path)
