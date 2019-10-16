import matplotlib.pyplot as plt

import torch
import numpy as np

from simulated_datasets_lib import plot_multiple_stars
import image_utils

def plot_image(fig, image,
                true_locs = None, estimated_locs = None,
                vmin = None, vmax = None,
                add_colorbar = False,
                global_fig = None,
                diverging_cmap = False,
                color = 'r', marker = 'x', alpha = 1):

    # locations are coordinates in the image, on scale from 0 to 1

    slen = image.shape[-1]

    if diverging_cmap:
        vmax = image.abs().max()
        im = fig.matshow(image, vmin = -vmax, vmax = vmax,
                            cmap=plt.get_cmap('bwr'))
    else:
        im = fig.matshow(image, vmin = vmin, vmax = vmax)

    if not(true_locs is None):
        assert len(true_locs.shape) == 2
        assert true_locs.shape[1] == 2
        fig.scatter(x = true_locs[:, 1] * (slen - 1),
                    y = true_locs[:, 0] * (slen - 1),
                    color = 'b')

    if not(estimated_locs is None):
        assert len(estimated_locs.shape) == 2
        assert estimated_locs.shape[1] == 2
        fig.scatter(x = estimated_locs[:, 1] * (slen - 1),
                    y = estimated_locs[:, 0] * (slen - 1),
                    color = color, marker = marker, alpha = alpha)

    if add_colorbar:
        assert global_fig is not None
        global_fig.colorbar(im, ax = fig)

def plot_categorical_probs(log_prob_vec, fig):
    n_cat = len(log_prob_vec)
    points = [(i, torch.exp(log_prob_vec[i])) for i in range(n_cat)]

    for pt in points:
        # plot (x,y) pairs.
        # vertical line: 2 x,y pairs: (a,0) and (a,b)
        fig.plot([pt[0],pt[0]], [0,pt[1]], color = 'blue')

    fig.plot(np.arange(n_cat),
             torch.exp(log_prob_vec).detach().numpy(),
             'o', markersize = 5, color = 'blue')

def plot_subimage(fig, full_image, full_est_locs, full_true_locs,
                    x0, x1, subimage_slen,
                    vmin = None, vmax = None,
                    add_colorbar = False,
                    global_fig = None,
                    diverging_cmap = False,
                    color = 'r', marker = 'x', alpha = 1):

    assert len(full_image.shape) == 2

    # full_est_locs and full_true_locs are locations in the coordinates of the
    # full image, in pixel units, scaled between 0 and 1


    # trim image to subimage
    image_patch = full_image[x0:(x0 + subimage_slen), x1:(x1 + subimage_slen)]

    # get locations in the subimage
    if full_est_locs is not None:
        assert torch.all(full_est_locs <= 1)
        assert torch.all(full_est_locs >= 0)

        _full_est_locs = full_est_locs * (full_image.shape[-1] - 1)

        which_est_locs = (_full_est_locs[:, 0] > x0) & \
                        (_full_est_locs[:, 0] < (x0 + subimage_slen - 1)) & \
                        (_full_est_locs[:, 1] > x1) & \
                        (_full_est_locs[:, 1] < (x1 + subimage_slen - 1))

        est_locs = (_full_est_locs[which_est_locs, :] - torch.Tensor([[x0, x1]])) / (subimage_slen - 1)
    else:
        est_locs = None


    if full_true_locs is not None:
        assert torch.all(full_true_locs <= 1)
        assert torch.all(full_true_locs >= 0)

        _full_true_locs = full_true_locs * (full_image.shape[-1] - 1)

        which_true_locs = (_full_true_locs[:, 0] > x0) & \
                        (_full_true_locs[:, 0] < (x0 + subimage_slen - 1)) & \
                        (_full_true_locs[:, 1] > x1) & \
                        (_full_true_locs[:, 1] < (x1 + subimage_slen - 1))

        true_locs = (_full_true_locs[which_true_locs, :] - torch.Tensor([[x0, x1]])) / (subimage_slen - 1)
    else:
        true_locs = None

    plot_image(fig, image_patch,
                    true_locs = true_locs,
                    estimated_locs = est_locs,
                    vmin = vmin, vmax = vmax,
                    add_colorbar = add_colorbar,
                    global_fig = global_fig,
                    diverging_cmap = diverging_cmap,
                    color = color, marker = marker, alpha = alpha)
