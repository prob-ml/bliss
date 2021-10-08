import galsim
import numpy as np
import torch
import tqdm
from einops import rearrange, reduce
from scipy import optimize as sp_optim
from sklearn.metrics import confusion_matrix


class MatchingMetrics:
    """Class that matches quantities based on source locations and calculates metrics.

    This clases is intended to be used for matching sources locations in a batch of images.
    """

    def __init__(
        self,
        true_locs: torch.Tensor,
        est_locs: torch.Tensor,
        true_n_sources: torch.Tensor,
        est_n_sources: torch.Tensor,
        slack=1.0,
    ) -> None:
        """Computes matches between true and estimated locations.

        Args:
            true_n_sources: Tensor of shape `(b,)` where `b` is the batch size.
            est_n_sources: Tensor of shape `(b,)` where `b` is the batch size.
            slack: Threshold for matching objects a `slack` l1 distance away (in pixels).
            true_locs: Tensor of shape `(b x n1 x 2)`, where `n1` is the true number of sources and
                `b` is the batch_size. The centroids should be in units of pixels.
            est_locs: Tensor of shape `(b x n2 x 2)`, where `n2` is the predicted
                number of sources and `b` is the batch_size. The centroids should be
                in units of pixels.

        Attributes:
           dist_keep: Matched objects to keep based on l1 distances.
           true_match, est_match: Tensors corresponding to indices matched, so
                that `true_locs[true_match]`, `est_locs[est_match]` will have the matched
                locations for each pair of matched objects at the same position.
        """
        assert len(true_locs.shape) == len(est_locs.shape) == 3
        assert true_locs.shape[-1] == est_locs.shape[-1] == 2
        assert len(true_n_sources.shape) == len(est_n_sources.shape) == 1
        assert true_n_sources.shape == est_n_sources.shape
        self.batch_size = true_locs.shape[0]

        self.true_locs = true_locs  # in pixels.
        self.est_locs = est_locs  # in pixels.
        self.true_n_sources = true_n_sources
        self.est_n_sources = est_n_sources
        self.slack = slack

        self.match_by_locs()

    def match(self, batch_index, *args):
        """Given pairs of true/est objects, return matched version of each."""
        matches = (self.match_true[batch_index], self.match_est[batch_index])
        dist_keep = self.dist_keep[batch_index]
        return (x[batch_index][matches[i % 2]][dist_keep] for i, x in enumerate(args))

    def match_by_locs(self):
        """Match true and estimated locations and returned indices to match.

        Permutes `est_locs` to find minimal error between `true_locs` and `est_locs`.
        The matching is done with `scipy.optimize.linear_sum_assignment`, which implements
        the Hungarian algorithm.
        """

        self.match_true = []
        self.match_est = []
        self.dist_keep = []
        self.distances = 0.0  # for metrics related to average distance of matched objects.

        for b in range(self.batch_size):

            n1 = self.true_n_sources[b].item()
            n2 = self.est_n_sources[b].item()
            locs1 = self.true_locs[b].view(-1, 2)
            locs2 = self.true_locs[b].view(-1, 2)

            # entry (i,j) is l1 distance between of ith loc in locs1 and the jth loc in locs2
            locs_err = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
            locs_err = reduce(locs_err, "i j k -> i j", "sum")

            # find minimal permutation and return matches
            row_indx, col_indx = sp_optim.linear_sum_assignment(locs_err.detach().cpu())

            assert len(row_indx) == len(col_indx) == min(n1, n2)

            # only match objects that satisfy threshold.
            dist = (locs1[row_indx] - locs2[col_indx]).abs().max(1)[0]
            dist_keep1 = (dist < self.slack).bool()

            self.match_true.append(row_indx)
            self.match_est.append(col_indx)
            self.dist_keep.append(dist_keep1)

    def get_detection_metrics(self):
        """Calculate precision, recall, f1-score."""
        tp = 0.0
        fp = 0.0
        total_true_n_sources = self.true_n_sources.sum().int().item()

        for b in range(self.batch_size):

            _, est_locs1 = self.match(b, self.true_locs, self.est_locs)

            # true positives = # of sources matched with a true source.
            # false positives = # of predicted sources not matched with true source
            tp1 = len(est_locs1)
            fp1 = self.est_n_sources[b].int().item() - len(est_locs1)
            assert fp1 >= 0
            tp += tp1
            fp += fp1

        precision = tp / (tp + fp)
        recall = tp / total_true_n_sources
        f1 = (2 * precision * recall) / (precision + recall)

        return {
            "precision": precision,  # = PPV = positive predictive value
            "recall": recall,  # = TPR = true positive rate
            "f1": f1,
            "tp": tp,  # true positives
            "fp": fp,  # false negatives
        }

    def get_classification_metrics(
        self, true_galaxy_bool: torch.Tensor, est_galaxy_bool: torch.Tensor
    ):
        """Calculate misclassification accuracy, and confusion matrix.

        Classification accuracy is evaluated only based on matched objects.

        Args:
            true_galaxy_bool: Tensor of true booleans of source is star or galaxy w/ shape (n1,)
            est_galaxy_bool: Tensor of pred. booleans of source is star or galaxy w/ shape (n2,)

        Returns:
            Dictionary containing metrics.
        """
        total_n_matches = 0.0
        total_correct_class = 0.0
        cf_matrix = np.zeros((2, 2))

        for b in range(self.batch_size):
            true_galaxy_bool1, est_galaxy_bool1 = self.match(b, true_galaxy_bool, est_galaxy_bool)

            total_n_matches += len(true_galaxy_bool1)
            total_correct_class += (true_galaxy_bool1 == est_galaxy_bool1).sum().int().item()

            cf_matrix += confusion_matrix(true_galaxy_bool1, est_galaxy_bool1)

        return {
            "class_acc": total_correct_class / total_n_matches,
            "confusion_matrix": cf_matrix,
        }

    # def get_l1_metrics(x_true, x_est)

    def get_l1_errors(self, *args):
        """Obtain L1 errors corresponding to pairs of true/est tensors matched by locations."""
        l1_errors = []
        for i in range(0, len(args), 2):  # only every other element
            true_x, est_x = self.match(args[i], args[i + 1])
            assert true_x.shape == est_x.shape
            assert len(true_x.shape) == 2
            l1_errors.append((true_x - est_x).abs().sum(1))
        return l1_errors


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
