"""Functions to evaluate the performance of BLISS predictions."""

import math
from collections import defaultdict
from copy import deepcopy
from typing import DefaultDict, Dict, Tuple

import galsim
import sep_pjw as sep
import torch
from einops import rearrange, reduce
from scipy import optimize as sp_optim
from torch import Tensor
from tqdm import tqdm

from bliss.catalog import FullCatalog
from bliss.datasets.lsst import BACKGROUND, PIXEL_SCALE
from bliss.encoders.autoencoder import CenteredGalaxyDecoder
from bliss.render_tiles import reconstruct_image_from_ptiles, render_galaxy_ptiles


def match_by_locs(true_locs, est_locs, slack=1.0):
    """Match true and estimated locations and returned indices to match.

    Permutes `est_locs` to find minimal error between `true_locs` and `est_locs`.
    The matching is done with `scipy.optimize.linear_sum_assignment`, which implements
    the Hungarian algorithm.

    Automatically discards matches where at least one location has coordinates **exactly** (0, 0).

    Args:
        slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
        true_locs: Tensor of shape `(n1 x 2)`, where `n1` is the true number of sources.
            The centroids should be in units of PIXELS.
        est_locs: Tensor of shape `(n2 x 2)`, where `n2` is the predicted
            number of sources. The centroids should be in units of PIXELS.

    Returns:
        A tuple of the following objects:
        - row_indx: Indicies of true objects matched to estimated objects.
        - col_indx: Indicies of estimated objects matched to true objects.
        - dist_keep: Matched objects to keep based on l1 distances.
        - avg_distance: Average l-infinity distance over matched objects.
    """
    assert len(true_locs.shape) == len(est_locs.shape) == 2
    assert true_locs.shape[-1] == est_locs.shape[-1] == 2
    assert isinstance(true_locs, torch.Tensor) and isinstance(est_locs, torch.Tensor)

    locs1 = true_locs.view(-1, 2)
    locs2 = est_locs.view(-1, 2)

    locs_abs_diff = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
    locs_err = reduce(locs_abs_diff, "i j k -> i j", "sum")
    locs_err_l_infty = reduce(locs_abs_diff, "i j k -> i j", "max")

    # Penalize all pairs which are greater than slack apart to favor valid matches.
    locs_err = locs_err + (locs_err_l_infty > slack) * locs_err.max()

    # find minimal permutation and return matches
    row_indx, col_indx = sp_optim.linear_sum_assignment(locs_err.detach().cpu())

    # we match objects based on distance too.
    # only match objects that satisfy threshold on l-infinity distance.
    # do not match fake objects with locs = (0, 0)
    dist = (locs1[row_indx] - locs2[col_indx]).abs().max(1)[0]
    origin_dist = torch.min(locs1[row_indx].pow(2).sum(1), locs2[col_indx].pow(2).sum(1))
    cond1 = (dist < slack).bool()
    cond2 = (origin_dist > 0).bool()
    dist_keep = torch.logical_and(cond1, cond2)
    avg_distance = dist[cond2].mean()  # average l-infinity distance over matched objects.
    if dist_keep.sum() > 0:
        assert dist[dist_keep].max() <= slack
    return row_indx, col_indx, dist_keep, avg_distance


def compute_batch_tp_fp(truth: FullCatalog, est: FullCatalog) -> Tuple[Tensor, Tensor, Tensor]:
    """Separate purpose from `DetectionMetrics`, since here we don't aggregate over batches."""
    all_tp = torch.zeros(truth.batch_size)
    all_fp = torch.zeros(truth.batch_size)
    all_ntrue = torch.zeros(truth.batch_size)
    for b in range(truth.batch_size):
        ntrue, nest = truth.n_sources[b].int().item(), est.n_sources[b].int().item()
        tlocs, elocs = truth.plocs[b], est.plocs[b]
        if ntrue > 0 and nest > 0:
            _, mest, dkeep, _ = match_by_locs(tlocs, elocs)
            tp = len(elocs[mest][dkeep])
            fp = nest - tp
        elif ntrue > 0:
            tp = 0
            fp = 0
        elif nest > 0:
            tp = 0
            fp = nest
        else:
            tp = 0
            fp = 0
        all_tp[b] = tp
        all_fp[b] = fp
        all_ntrue[b] = ntrue

    return all_tp, all_fp, all_ntrue


def compute_tp_fp_per_bin(
    truth: FullCatalog,
    est: FullCatalog,
    param: str,
    bins: Tensor,
) -> Dict[str, Tensor]:
    counts_per_bin: DefaultDict[str, Tensor] = defaultdict(
        lambda: torch.zeros(len(bins), truth.batch_size)
    )
    for ii, (b1, b2) in tqdm(enumerate(bins), desc="tp/fp per bin", total=len(bins)):
        # precision
        eparams = est.apply_param_bin(param, b1, b2)
        tp, fp, _ = compute_batch_tp_fp(truth, eparams)
        counts_per_bin["tp_precision"][ii] = tp
        counts_per_bin["fp_precision"][ii] = fp

        # recall
        tparams = truth.apply_param_bin(param, b1, b2)
        tp, _, ntrue = compute_batch_tp_fp(tparams, est)
        counts_per_bin["tp_recall"][ii] = tp
        counts_per_bin["ntrue"][ii] = ntrue

    return counts_per_bin


def get_boostrap_precision_and_recall(
    n_samples: int,
    truth: FullCatalog,
    est: FullCatalog,
    param: str,
    bins: Tensor,
) -> Dict[str, Tensor]:
    """Get errors for precision/recall which need to be handled carefully to be efficient."""
    counts_per_bin = compute_tp_fp_per_bin(truth, est, param, bins)
    batch_size = truth.batch_size
    n_bins = bins.shape[0]

    # get counts in needed format
    tpp_boot = counts_per_bin["tp_precision"].unsqueeze(0).expand(n_samples, n_bins, batch_size)
    fpp_boot = counts_per_bin["fp_precision"].unsqueeze(0).expand(n_samples, n_bins, batch_size)
    tpr_boot = counts_per_bin["tp_recall"].unsqueeze(0).expand(n_samples, n_bins, batch_size)
    ntrue_boot = counts_per_bin["ntrue"].unsqueeze(0).expand(n_samples, n_bins, batch_size)

    # get indices to boostrap
    # NOTE: the indices for each sample repeat across bins
    boot_indices = torch.randint(0, batch_size, (n_samples, 1, batch_size))
    boot_indices = boot_indices.expand(n_samples, n_bins, batch_size)

    # get bootstrapped samples of counts
    tpp_boot = torch.gather(tpp_boot, 2, boot_indices)
    fpp_boot = torch.gather(fpp_boot, 2, boot_indices)
    tpr_boot = torch.gather(tpr_boot, 2, boot_indices)
    ntrue_boot = torch.gather(ntrue_boot, 2, boot_indices)
    assert tpp_boot.shape == (n_samples, n_bins, batch_size)

    # finally, get precision and recall boostrapped samples
    precision_boot = tpp_boot.sum(2) / (tpp_boot.sum(2) + fpp_boot.sum(2))
    recall_boot = tpr_boot.sum(2) / ntrue_boot.sum(2)

    assert precision_boot.shape == (n_samples, n_bins)
    assert recall_boot.shape == (n_samples, n_bins)
    return {"precision": precision_boot, "recall": recall_boot}


def get_single_galaxy_ellipticities(images: Tensor, no_bar: bool = True) -> Tensor:
    """Returns ellipticities of (noiseless, single-band) individual galaxy images.

    Args:
        images: Array of shape (n, slen, slen) containing single galaxies with no noise/background.
        no_bar: Whether to use a progress bar.

    Returns:
        Tensor containing ellipticity measurements for each galaxy in `images`.
    """
    assert images.device == torch.device("cpu")
    assert images.ndim == 3
    n_samples, _, _ = images.shape
    ellips = torch.zeros((n_samples, 2))  # 2nd shape: e1, e2
    images_np = images.numpy()

    # Now we use galsim to measure size and ellipticity
    for ii in tqdm(range(n_samples), desc="Measuring galaxies", disable=no_bar):
        image = images_np[ii]
        if image.sum() > 0:  # skip empty images
            galsim_image = galsim.Image(image, scale=PIXEL_SCALE)
            # sigma ~ size of psf (in pixels)
            out = galsim.hsm.FindAdaptiveMom(galsim_image, guess_sig=3, strict=False)
            if out.error_message == "":
                e1 = float(out.observed_e1)
                e2 = float(out.observed_e2)
            else:
                e1, e2 = float("nan"), float("nan")  # noqa: WPS456
            ellips[ii, :] = torch.tensor([e1, e2])
    return ellips


def get_snr(noiseless: Tensor) -> Tensor:
    """Compute SNR given noiseless, isolated iamges of galaxies and background."""
    image_with_background = noiseless + BACKGROUND
    snr2 = reduce(noiseless**2 / image_with_background, "b c h w -> b", "sum")
    return torch.sqrt(snr2)


def get_blendedness(iso_image: Tensor, blend_image: Tensor) -> Tensor:
    """Calculate blendedness.

    Args:
        iso_image: Array of shape = (B, N, C, H, W) corresponding to images of the isolated
            galaxy you are calculating blendedness for (noiseless)

        blend_image: Array of shape = (B, C, H, W) corresponding to the blended image (noiseless).

    """
    assert iso_image.ndim == 5
    num = reduce(iso_image * iso_image, "b s c h w -> b s", "sum")
    blend = rearrange(blend_image, "b c h w -> b 1 c h w")
    denom = reduce(blend * iso_image, "b s c h w -> b s", "sum")
    blendedness = 1 - num / denom
    return torch.where(blendedness.isnan(), 0, blendedness)


def get_deblended_reconstructions(
    cat: FullCatalog,
    dec: CenteredGalaxyDecoder,
    *,
    slen: int,
    device: torch.device,
    batch_size: int = 100,
    ptile_slen: int = 52,
    bp: int = 24,
    tile_slen: int = 4,
    no_bar: bool = True,
):
    """Return deblended galaxy reconstructions on prediction locations on a full size stamp."""

    n_images = cat.batch_size
    n_batches = math.ceil(n_images / batch_size)
    image_size = slen + 2 * bp
    recon_uncentered = torch.zeros((n_images, cat.max_n_sources, 1, image_size, image_size))

    for jj in tqdm(range(cat.max_n_sources), desc="Obtaining reconstructions", disable=no_bar):
        mask = torch.arange(cat.max_n_sources)
        mask = mask[mask != jj]

        # make a copy with all except one galaxy zeroed out
        est_jj = FullCatalog(slen, slen, deepcopy(cat.to_dict()))
        est_jj["galaxy_bools"][:, mask, :] = 0
        est_jj["galaxy_bools"] = est_jj["galaxy_bools"].contiguous()

        # will fail if catalog does nto come from encoder(s)
        est_tiled_jj = est_jj.to_tile_params(tile_slen, ignore_extra_sources=False)

        images_jj = []
        for kk in range(n_batches):
            start, end = kk * batch_size, (kk + 1) * batch_size
            blocs = est_tiled_jj.locs[start:end].to(device)
            bgparams = est_tiled_jj["galaxy_params"][start:end].to(device)
            bgbools = est_tiled_jj["galaxy_bools"][start:end].to(device)

            galaxy_tiles = render_galaxy_ptiles(
                dec,
                locs=blocs,
                galaxy_params=bgparams,
                galaxy_bools=bgbools,
                ptile_slen=ptile_slen,
                tile_slen=tile_slen,
            )

            galaxy_images = reconstruct_image_from_ptiles(galaxy_tiles, tile_slen)
            images_jj.append(galaxy_images.cpu())

        images_jj = torch.concatenate(images_jj, axis=0)
        recon_uncentered[:, jj, :, :, :] = images_jj

    return recon_uncentered


def get_residual_measurements(
    cat: FullCatalog,
    images: Tensor,
    *,
    paddings: Tensor,
    sources: Tensor,
    bp: int = 24,
    r: float = 5.0,
    no_bar: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Obtain aperture photometry fluxes for each source in the catalog."""
    n_batches = cat.n_sources.shape[0]

    fluxes = torch.zeros(n_batches, cat.max_n_sources, 1)
    fluxerrs = torch.zeros(n_batches, cat.max_n_sources, 1)
    snrs = torch.zeros(n_batches, cat.max_n_sources, 1)

    ellips = torch.zeros(n_batches, cat.max_n_sources, 2)
    sigmas = torch.zeros(n_batches, cat.max_n_sources, 1)

    for ii in tqdm(range(n_batches), desc="Measuring galaxies", disable=no_bar):
        n_sources = cat.n_sources[ii].item()
        y = cat.plocs[ii, :, 0].numpy() + bp - 0.5
        x = cat.plocs[ii, :, 1].numpy() + bp - 0.5

        # obtain residual images for each galaxy to measure SNR
        image = images[ii, 0] - paddings[ii, 0]
        each_galaxy = sources[ii, :, 0]  # n h w
        all_galaxies = rearrange(reduce(each_galaxy, "n h w -> h w", "sum"), "h w -> 1 h w")
        other_galaxies = all_galaxies - each_galaxy
        residual_images = rearrange(image, "h w -> 1 h w") - other_galaxies

        _fluxes = []
        _fluxerrs = []
        _e1s = []
        _e2s = []
        _sigmas = []

        for jj in range(n_sources):
            target_img = residual_images[jj].numpy()

            # measure fluxes with sep
            f, ferr, _ = sep.sum_circle(
                target_img, [x[jj]], [y[jj]], r=r, err=BACKGROUND.sqrt().item()
            )
            _fluxes.append(f.item())
            _fluxerrs.append(ferr.item())

            # measure ellipticities and size with adaptive moments
            _galsim_img = galsim.Image(target_img, scale=PIXEL_SCALE)
            _centroid = galsim.PositionD(x=x[jj] + 1, y=y[jj] + 1)
            out = galsim.hsm.FindAdaptiveMom(
                _galsim_img, guess_centroid=_centroid, guess_sig=3, strict=False
            )
            if out.error_message == "":
                e1 = float(out.observed_e1)
                e2 = float(out.observed_e2)
                sigma = float(out.moments_sigma)
            else:
                e1 = float("nan")  # noqa: WPS456
                e2 = float("nan")  # noqa: WPS456
                sigma = float("nan")  # noqa: WPS456

            _e1s.append(e1)
            _e2s.append(e2)
            _sigmas.append(sigma)

        fluxes[ii, :n_sources, 0] = torch.tensor(_fluxes)
        fluxerrs[ii, :n_sources, 0] = torch.tensor(_fluxerrs)
        snrs[ii, :n_sources, 0] = fluxes[ii, :n_sources, 0] / fluxerrs[ii, :n_sources, 0]

        ellips[ii, :n_sources, 0] = torch.tensor(_e1s)
        ellips[ii, :n_sources, 1] = torch.tensor(_e2s)
        sigmas[ii, :n_sources, 0] = torch.tensor(_sigmas)

    return {"flux": fluxes, "fluxerr": fluxerrs, "snr": snrs, "ellips": ellips, "sigma": sigmas}
