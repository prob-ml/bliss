import galsim
import numpy as np
import torch
import tqdm
from einops import rearrange, reduce
from scipy import optimize as sp_optim
from sklearn.metrics import confusion_matrix

# TODO: Need to check that when calculating tpr/ppv we use correct total # of objects by
# `true_n_sources`.


class MatchingMetrics:
    """Class that matches quantities based on source locations and calculated metrics."""

    def __init__(
        self,
        true_n_sources: torch.Tensor,
        est_n_sources: torch.Tensor,
        true_locs: torch.Tensor,
        est_locs: torch.Tensor,
        slack=1.0,
    ) -> None:
        """Computes matches between true and estimated locations.

        Args:
            true_n_sources: FIXME.
            est_n_sources: FIXME.
            true_locs: Tensor of shape `(n1 x 2)`, where `n1` is the number of sources (in pixels).
            est_locs: Tensor of shape `(n2 x 2)`, where `n2` is the number of sources (in pixels).
            slack: Threshold for matching objects a `slack` l1 distance away (in pixels).

        Attributes:
           dist_keep: Matched objects to keep based on l1 distances.
           true_match, est_match: Tensors corresponding to indices matched, so
                that `true_locs[true_match]`, `est_locs[est_match]` will have the matched
                locations for each pair of matched objects at the same position.
        """
        assert len(true_locs.shape) == len(est_locs.shape) == 2
        assert true_locs.shape[1] == est_locs.shape[1] == 2

        self.true_n_sources = true_n_sources
        self.est_n_sources = est_n_sources
        self.true_locs = true_locs  # in pixels.
        self.est_locs = est_locs  # in pixels.
        self.slack = slack

        # match using `match_by_locs`
        true_match, est_match, dist_keep = self.match_locs()
        self.true_match = true_match
        self.est_match = est_match
        self.dist_keep = dist_keep

        self.true_n_total = len(self.true_locs)  # total true number of objects.

    def get_matchings(self):
        """Get information to performa matching manually."""
        return self.true_match, self.est_match, self.dist_keep

    def get_matched(self, *args):
        """Given pairs of true/est objects, return matched version of each."""
        matches = (self.true_match, self.est_match)
        return (x[matches[i % 2]][self.dist_keep] for i, x in enumerate(args))

    def match_locs(self):
        """Match true and estimated locations and returned indices to match.

        Permutes `est_locs` to find minimal error between `true_locs` and `est_locs`.
        The matching is done with `scipy.optimize.linear_sum_assignment`, which implements
        the Hungarian algorithm.

        Returns:
            Tensors to do the matching.
        """
        locs1, locs2 = self.true_locs, self.est_locs
        assert len(locs1.shape) == len(locs2.shape) == 2
        assert locs1.shape[1] == locs2.shape[1] == 2

        # entry (i,j) is l1 distance between of ith loc in locs1 and the jth loc in locs2
        locs_err = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
        locs_err = reduce(locs_err, "i j k -> i j", "sum")

        # find minimal permutation and return matches
        row_indx, col_indx = sp_optim.linear_sum_assignment(locs_err.detach().cpu())

        # only match objects that satisfy threshold.
        dist = (locs1[row_indx] - locs2[col_indx]).abs().max(1)[0]
        dist_keep = (dist < self.slack).bool()

        return row_indx, col_indx, dist_keep

    def get_detect_class_metrics(
        self, true_galaxy_bool: torch.Tensor, est_galaxy_bool: torch.Tensor
    ):
        """Calculate precision, recall, f1-score, misclassification accuracy, and confusion matrix.

        The misclassfication metrics are calculated based only on matched objects.

        Args:
            true_galaxy_bool: Tensor of true booleans of source is star or galaxy w/ shape (n1,)
            est_galaxy_bool: Tensor of pred. booleans of source is star or galaxy w/ shape (n2,)

        Returns:
            Dictionary containing metrics.
        """
        # 1 = 'matched'
        true_locs1, est_locs1 = self.get_matched(self.true_locs, self.est_locs)
        true_galaxy_bool1, est_galaxy_bool1 = self.get_matched(true_galaxy_bool, est_galaxy_bool)
        assert len(true_locs1) == len(est_locs1)
        assert len(true_galaxy_bool1) == len(est_galaxy_bool1)

        # true positives = # of sources matched with a true source.
        # false positives = # of predicted sources not matched with true source
        tp = len(est_locs1)
        fp = len(self.est_locs) - len(est_locs1)
        assert fp >= 0

        precision = tp / (tp + fp)
        recall = tp / self.true_n_total
        f1 = (2 * precision * recall) / (precision + recall)

        # number of matched, misclassified objects
        ratio_misclassed = sum(true_galaxy_bool1 != est_galaxy_bool1) / len(true_galaxy_bool1)
        cf_matrix = confusion_matrix(true_galaxy_bool1, est_galaxy_bool1, labels=[True, False])

        return {
            "precision": precision,  # = PPV = positive predictive value
            "recall": recall,  # = TPR = true positive rate
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "class_acc": ratio_misclassed,
            "confusion_matrix": cf_matrix,
        }

    def get_l1_errors(self, *args):
        """Obtain L1 errors corresponding to pairs of true/est tensors matched by locations."""
        l1_errors = []
        for i in range(0, len(args), 2):  # only every other element
            true_x, est_x = self.get_matched(args[i], args[i + 1])
            assert true_x.shape == est_x.shape
            assert len(true_x.shape) == 2
            l1_errors.append((true_x - est_x).abs().sum(1))
        return l1_errors


def get_metrics_on_batch(true_params: dict, est_params: dict, slen: int):
    """Return error vectors and metrics for parameters on batches. The parameters are not on tiles.

    Args:
        slen: Side length of image.
        true_params: Dictionary containing true parameters on whole image on batches. It must
            contain at least keys: ('n_sources', 'fluxes', 'locs', 'galaxy_bool', 'galaxy_params').
            Each parameter must be of shape (b, n1, ...) where b = batch size, n1 = max sources.
        est_params: Dictionary containing estimated parameters on whole image on batches. It must
            contain at least keys: ('n_sources', 'fluxes', 'locs', 'galaxy_bool', 'galaxy_params').
            Each parameter must be of shape (b, n2, ...) where b = batch size, n2 = max sources.

    Returns:
        Dictionary of metrics.
    """
    assert len(true_params["n_sources"]) == len(est_params["n_sources"]), "Batch sizes differ."

    batch_size = len(true_params["n_sources"])
    n_bands = true_params["fluxes"].shape[-1]

    tpr_vec = torch.zeros(batch_size)
    ppv_vec = torch.zeros(batch_size)

    locs_mae_vec = []
    fluxes_mae_vec = []
    galaxy_params_mae_vec = []

    # boolean accuracy of counting number of sources
    count_bool = true_params["n_sources"].eq(est_params["n_sources"])

    # accuracy of galaxy counts
    est_n_gal = reduce(est_params["galaxy_bool"], "b n 1 -> b", "sum")
    true_n_gal = reduce(true_params["galaxy_bool"], "b n 1 -> b", "sum")
    galaxy_counts_bool = est_n_gal.eq(true_n_gal)

    # TODO: Need to calculate l1 vectors, also account for the fact that sometimes batches have
    # no objects in them so averaging might be off.
    for i in range(batch_size):

        # get number of sources
        ntrue = int(true_params["n_sources"][i])
        nest = int(est_params["n_sources"][i])

        if (nest > 0) and (ntrue > 0):

            # prepare locs and get them in units of pixels.
            _ = true_params["locs"][i, 0:ntrue].view(ntrue, 2) * slen
            _ = est_params["locs"][i, 0:nest].view(nest, 2) * slen

            # prepare fluxes
            _ = true_params["fluxes"][i, 0:ntrue].view(ntrue, n_bands)
            _ = est_params["fluxes"][i, 0:nest].view(nest, n_bands)

            # prepare galaxy params.
            _ = true_params["galaxy_params"][i, 0:ntrue].view(ntrue, -1)
            _ = est_params["galaxy_params"][i, 0:nest].view(nest, -1)

    locs_mae_vec = torch.tensor(locs_mae_vec)
    fluxes_mae_vec = torch.tensor(fluxes_mae_vec)
    galaxy_params_mae_vec = torch.tensor(galaxy_params_mae_vec)

    return {
        "locs_mae_vec": locs_mae_vec,
        "fluxes_mae_vec": fluxes_mae_vec,
        "galaxy_params_mae_vec": galaxy_params_mae_vec,
        "count_bool": count_bool,
        "galaxy_counts_bool": galaxy_counts_bool,
        "tpr_vec": tpr_vec,
        "ppv_vec": ppv_vec,
    }


def get_single_galaxy_measurements(
    slen: int,
    true_images: np.ndarray,
    recon_images: np.ndarray,
    psf_image: np.ndarray,
    pixel_scale: float = 0.396,
):
    """Compute individual galaxy measurements comparing true images with reconstructed images.

    Args:
        slen: Side-length of square input images.
        pixel_scale: Conversion from arcseconds to pixel.
        true_images: Array of shape (n_samples, n_bands, slen, slen) containing images of
            single-centered galaxies without noise or background.
        recon_images: Array of shape (n_samples, n_bands, slen, slen) containing
            reconstructions of `true_images` without noise or background.
        psf_image: Array of shape (n_bands, slen, slen) containing PSF image used for
            convolving the galaxies in `true_images`.

    Returns:
        Dictionary containing second-moment measurements for `true_images` and `recon_images`.
    """
    # TODO: Consider multiprocessing? (if necessary)
    assert true_images.shape == recon_images.shape
    assert len(true_images.shape) == len(recon_images.shape) == 4, "Incorrect array format."
    assert true_images.shape[1] == recon_images.shape[1] == psf_image.shape[0] == 1  # one band
    n_samples = true_images.shape[0]
    true_images = true_images.reshape(-1, slen, slen)
    recon_images = recon_images.reshape(-1, slen, slen)
    psf_image = psf_image.reshape(slen, slen)

    true_fluxes = true_images.sum(axis=(1, 2))
    recon_fluxes = recon_images.sum(axis=(1, 2))

    true_hlrs = np.zeros((n_samples))
    recon_hlrs = np.zeros((n_samples))
    true_ellip = np.zeros((n_samples, 2))  # 2nd shape: e1, e2
    recon_ellip = np.zeros((n_samples, 2))

    # get galsim PSF
    galsim_psf_image = galsim.Image(psf_image, scale=pixel_scale)

    # Now we use galsim to measure size and ellipticity
    for i in tqdm.tqdm(range(n_samples)):
        true_image = true_images[i]
        recon_image = recon_images[i]

        galsim_true_image = galsim.Image(true_image, scale=pixel_scale)
        galsim_recon_image = galsim.Image(recon_image, scale=pixel_scale)

        true_hlrs[i] = galsim_true_image.calculateHLR()  # PSF-convolved.
        recon_hlrs[i] = galsim_recon_image.calculateHLR()

        res_true = galsim.hsm.EstimateShear(
            galsim_true_image, galsim_psf_image, shear_est="KSB", strict=False
        )
        res_recon = galsim.hsm.EstimateShear(
            galsim_recon_image, galsim_psf_image, shear_est="KSB", strict=False
        )

        true_ellip[i, :] = (res_true.corrected_g1, res_true.corrected_g2)
        recon_ellip[i, :] = (res_recon.corrected_g1, res_recon.corrected_g2)

    return {
        "true_fluxes": true_fluxes,
        "recon_fluxes": recon_fluxes,
        "true_ellip": true_ellip,
        "recon_ellip": recon_ellip,
        "true_hlrs": true_hlrs,
        "recon_hlrs": recon_hlrs,
    }
