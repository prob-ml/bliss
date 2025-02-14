import flexcode  # pylint: disable=import-error
import numpy as np
import qp  # pylint: disable=import-error


def compute_gridded_pdfs(predictions: qp.ensemble.Ensemble, n_gridpoints=1800):
    """Computes gridded pdfs from a qp.ensemble.Ensemble object.

    Args:
        predictions: pdf estimate for many samples
        n_gridpoints: Number of gridpoints to use for the gridded pdfs

    Returns:
        xs: np.array of shape (n_gridpoints,)
        gridded_pdfs: np.array of shape (n_points,n_gridpoints
    """

    dist = predictions.dist
    weights = dist._weights  # pylint: disable=protected-access # noqa: WPS437
    basis_coefficients = dist._basis_coefficients  # pylint: disable=protected-access # noqa: WPS437
    _, n_basis = weights.shape
    basis = flexcode.basis_functions.cosine_basis(np.r_[0 : 1 : n_gridpoints * 1j], n_basis)

    gridded_pdfs = weights @ basis.T

    for pdf in gridded_pdfs:
        flexcode.post_processing.normalize(pdf)
        flexcode.post_processing.remove_bumps(pdf, basis_coefficients.bump_threshold)
        flexcode.post_processing.sharpen(pdf, basis_coefficients.sharpen_alpha)

    # make bona fide pdfs on the actual scale of interest
    normalizers = np.mean(gridded_pdfs, axis=1, keepdims=True) * (dist.z_max - dist.z_min)
    gridded_pdfs /= normalizers
    xs = np.r_[dist.z_min : dist.z_max : n_gridpoints * 1j]

    return xs, gridded_pdfs


def compute_means(xs, gridded_pdfs):
    """Computes the mean of the gridded pdfs."""
    return np.sum(xs * gridded_pdfs, axis=1)


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


class MeanAbsolusteLoss:
    """Mean absolute loss function."""

    def __call__(self, y, yhat):
        return np.abs(y - yhat)


def estimate_loss_minimizers(xs, gridded_pdfs, loss_func):
    """For each point, identify a prediction that minimizes the expected loss.

    Args:
        xs: np.array of shape (n_gridpoints,)
        gridded_pdfs: np.array of shape (n_points,n_gridpoints)
        loss_func: ufunc of the form loss_func(y,yhat) -> float

    Returns:
        np.array of shape (n_points,) containing the minimizer for each point.
    """

    # losses[i,j] is the total when truth is xs[i] but we choose xs[j]
    # sum over possible truths to get expected loss for each guess
    # get best choice for each point
    losses = loss_func(xs[:, None], xs[None, :])
    loss_if_chosen = gridded_pdfs @ losses
    return xs[np.argmin(loss_if_chosen, axis=1)]
