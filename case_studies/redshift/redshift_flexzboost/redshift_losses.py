import numpy as np


class CatastrophicOutlierLoss:
    """True if abs(y-yhat) > catastrophe_threshold, and False otherwise."""

    def __init__(self, catastrophe_threshold=1):
        self.catastrophe_threshold = catastrophe_threshold

    def __call__(self, y, yhat):
        return np.abs(y - yhat) > self.catastrophe_threshold


class OnePlusOutlierLoss:
    """True if the abs(y-yhat) / (1+y) exceeds .15, and False otherwise.

    Salvato, M., et al. "Dissecting photometric redshift for active
    galactic nucleus using XMM-and Chandra-COSMOS samples."
    The Astrophysical Journal 742.2 (2011): 61.
    """

    def __call__(self, y, yhat):
        return np.abs(y - yhat) / (1 + y) > 0.15


class MeanSquaredLoss:
    """Mean squared loss function."""

    def __call__(self, y, yhat):
        return (y - yhat) ** 2


class MeanAbsoluteLoss:
    """Mean absolute loss function."""

    def __call__(self, y, yhat):
        return np.abs(y - yhat)


def estimate_loss_minimizers(xs, gridded_pdfs, loss_func):
    """For each point, identify a prediction that minimizes the expected loss.

    Args:
        xs: np.array of shape (n_gridpoints,), assumed evenly spaced
        gridded_pdfs: np.array of shape (n_points,n_gridpoints)
        loss_func: ufunc of the form loss_func(y,yhat) -> float

    Returns:
        np.array of shape (n_points,) containing the minimizer value of xs
         (not integer indexing into  xs, but a float value from xs) for each point.
    """

    # losses[i,j] is the total when truth is xs[i] but we choose xs[j]
    # sum over possible truths to get expected loss for each guess
    # get best choice for each point
    losses = loss_func(xs[:, None], xs[None, :])
    loss_if_chosen = gridded_pdfs @ losses

    # loss_if_chosen[i,j] is the expected loss if we choose xs[j] for the ith point
    # so we want to minimize over j
    # BUT in some cases there may be multiple xs[j] that minimize the loss
    # so we take average of all such values
    results = []
    for loss_row in loss_if_chosen:
        indices = np.where(loss_row == loss_row.min())[0]
        middle_index = indices[len(indices) // 2]
        results.append(xs[middle_index])

    return np.array(results)
