"""Functions to evaluate the performance of BLISS predictions."""

import math
from collections import defaultdict
from copy import deepcopy
from typing import Callable, DefaultDict, Dict, Tuple

import galsim
import numpy as np
import sep
import torch
from einops import rearrange, reduce
from scipy import optimize as sp_optim
from torch import Tensor
from tqdm import tqdm

from bliss.catalog import FullCatalog, collate
from bliss.datasets.lsst import APERTURE_BACKGROUND, BACKGROUND, PIXEL_SCALE
from bliss.encoders.autoencoder import CenteredGalaxyDecoder
from bliss.render_tiles import reconstruct_image_from_ptiles, render_galaxy_ptiles


def match_by_locs(locs1: Tensor, locs2: Tensor, *, slack: float = 2.0):
    """Match true and estimated locations and returned indices to match.

    Automatically discards matches where at least one location has coordinates **exactly** (0, 0).

    Args:
        slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
        locs1: Tensor of shape `(n1 x 2)`, where `n1` is the true number of sources.
            The centroids should be in units of PIXELS.
        locs2: Tensor of shape `(n2 x 2)`, where `n2` is the predicted
            number of sources. The centroids should be in units of PIXELS.

    Returns:
        A tuple of the following objects:
        - row_indx: Indicies of true objects matched to estimated objects.
        - col_indx: Indicies of estimated objects matched to true objects.
        - dist_keep: Matched objects to keep based on l1 distances.
        - avg_distance: Average l-infinity distance over matched objects.
    """
    assert locs1.ndim == locs2.ndim == 2
    assert locs1.shape[-1] == locs2.shape[-1] == 2

    locs_diff = rearrange(locs1, "i xy -> i 1 xy") - rearrange(locs2, "i xy -> 1 i xy")
    dist1 = reduce(locs_diff.pow(2), "i j xy -> i j", "sum").sqrt()
    err = torch.where(dist1 > slack, dist1.max() * 100, dist1)

    # find minimal permutation and return matches
    row_indx, col_indx = sp_optim.linear_sum_assignment(err.detach().cpu())

    # drop matches with distance greater than slack
    # do not match fake objects with locs = (0, 0) exactly
    dist2 = (locs1[row_indx] - locs2[col_indx]).pow(2).sum(1).sqrt()
    origin_dist = torch.min(locs1[row_indx].pow(2).sum(1), locs2[col_indx].pow(2).sum(1))
    cond1 = (dist2 < slack).bool()
    cond2 = (origin_dist > 0).bool()
    dist_keep = torch.logical_and(cond1, cond2)
    avg_distance = dist2[cond2].mean()  # average distance over matched objects.
    if dist_keep.sum() > 0:
        assert dist2[dist_keep].max() <= slack
    return row_indx, col_indx, dist_keep, avg_distance


def match_by_grade(
    *,
    locs1: Tensor,
    locs2: Tensor,
    fluxes1: Tensor,
    fluxes2: Tensor,
    slack: float = 2.0,
    background: float = APERTURE_BACKGROUND,
):
    """Match objects based on centroids and flux."""
    assert locs1.ndim == locs2.ndim == 2
    assert locs1.shape[-1] == locs2.shape[-1] == 2
    assert fluxes1.ndim == fluxes2.ndim == 1
    assert torch.all((fluxes1 >= 0) | (fluxes1.abs() < APERTURE_BACKGROUND))
    assert isinstance(locs1, torch.Tensor) and isinstance(locs2, torch.Tensor)
    assert isinstance(fluxes1, torch.Tensor) and isinstance(fluxes2, torch.Tensor)

    locs_diff = rearrange(locs1, "i xy -> i 1 xy") - rearrange(locs2, "i xy -> 1 i xy")
    flux_diff = rearrange(fluxes1, "i -> i 1") - rearrange(fluxes2, "i -> 1 i")

    dist1 = reduce(locs_diff.pow(2), "i j xy -> i j", "sum").sqrt()
    inv_grade = 1 + flux_diff.abs() / (rearrange(fluxes1, "b -> b 1") + background)
    err = torch.where(dist1 > slack, inv_grade.max() * 100, inv_grade)
    row_indx, col_indx = sp_optim.linear_sum_assignment(err.detach().cpu())

    # we match objects based on distance too.
    # only match objects that satisfy threshold on l-infinity distance.
    # do not match fake objects with locs = (0, 0) exactly
    dist2 = (locs1[row_indx] - locs2[col_indx]).pow(2).sum(1).sqrt()
    origin_dist = torch.min(locs1[row_indx].pow(2).sum(1), locs2[col_indx].pow(2).sum(1))
    cond1 = (dist2 < slack).bool()
    cond2 = (origin_dist > 0).bool()
    dist_keep = torch.logical_and(cond1, cond2)
    avg_distance = dist2[cond2].mean()
    if dist_keep.sum() > 0:
        assert dist2[dist_keep].max() <= slack
    return row_indx, col_indx, dist_keep, avg_distance


def compute_batch_tp_fp(
    truth: FullCatalog,
    est: FullCatalog,
    *,
    slack: float = 2.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Separate purpose from `DetectionMetrics`, since here we don't aggregate over batches."""
    all_tp = torch.zeros(truth.batch_size)
    all_fp = torch.zeros(truth.batch_size)
    all_ntrue = torch.zeros(truth.batch_size)
    for b in range(truth.batch_size):
        ntrue, nest = truth.n_sources[b].int().item(), est.n_sources[b].int().item()
        tlocs, elocs = truth.plocs[b], est.plocs[b]
        if ntrue > 0 and nest > 0:
            _, mest, dkeep, _ = match_by_locs(tlocs, elocs, slack=slack)
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
    *,
    slack: float = 2.0,
    only_recall: bool = False,
) -> Dict[str, Tensor]:
    counts_per_bin: DefaultDict[str, Tensor] = defaultdict(
        lambda: torch.zeros(len(bins), truth.batch_size)
    )
    for ii, (b1, b2) in tqdm(enumerate(bins), desc="tp/fp per bin", total=len(bins)):
        if not only_recall:
            # precision
            eparams = est.apply_param_bin(param, b1, b2)
            tp, fp, _ = compute_batch_tp_fp(truth, eparams, slack=slack)
            counts_per_bin["tp_precision"][ii] = tp
            counts_per_bin["fp_precision"][ii] = fp

        # recall
        tparams = truth.apply_param_bin(param, b1, b2)
        tp, _, ntrue = compute_batch_tp_fp(tparams, est, slack=slack)
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
            if not out.error_message:
                e1 = float(out.observed_e1)
                e2 = float(out.observed_e2)
            else:
                e1, e2 = float("nan"), float("nan")
            ellips[ii, :] = torch.tensor([e1, e2])
    return ellips


def get_snr(noiseless: Tensor) -> Tensor:
    """Compute SNR given noiseless, isolated images of galaxies and background."""
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
    bp: int = 24,
    tile_slen: int = 5,
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
                tile_slen=tile_slen,
            )

            galaxy_images = reconstruct_image_from_ptiles(galaxy_tiles, tile_slen)
            images_jj.append(galaxy_images.cpu())

        images_jj = torch.concatenate(images_jj, axis=0)
        recon_uncentered[:, jj, :, :, :] = images_jj

    return recon_uncentered.to("cpu").contiguous()


def get_residual_measurements(
    cat: FullCatalog,
    images: Tensor,
    *,
    paddings: Tensor,
    sources: Tensor,
    bp: int = 24,
    r: float = 5.0,
    no_bar: bool = True,
) -> dict[Tensor]:
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
                target_img, [x[jj]], [y[jj]], r=r, err=BACKGROUND.sqrt().item(), gain=1.0
            )
            _fluxes.append(f.item())
            _fluxerrs.append(ferr.item())

            # measure ellipticities and size with adaptive moments
            _galsim_img = galsim.Image(target_img, scale=PIXEL_SCALE)
            _centroid = galsim.PositionD(x=x[jj] + 1, y=y[jj] + 1)
            out = galsim.hsm.FindAdaptiveMom(
                _galsim_img, guess_centroid=_centroid, guess_sig=3, strict=False
            )
            if not out.error_message:
                e1 = float(out.observed_e1)
                e2 = float(out.observed_e2)
                sigma = float(out.moments_sigma)
            else:
                e1 = float("nan")
                e2 = float("nan")
                sigma = float("nan")

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


def pred_in_batches(
    pred_fnc: Callable,
    images: Tensor,
    *args,
    device: torch.device,
    batch_size: int = 200,
    desc: str = "",
    no_bar: bool = True,
    axis=0,
):
    # gotta ensure model.forward outputs a dict of Tensors
    n_images = images.shape[0]
    n_batches = math.ceil(n_images / batch_size)
    tiled_params_list = []
    with torch.no_grad():
        for ii in tqdm(range(n_batches), desc=desc, disable=no_bar):
            start, end = ii * batch_size, (ii + 1) * batch_size
            image_batch = images[start:end].to(device)
            args_batch = (x[start:end].to(device) for x in args)
            d = pred_fnc(image_batch, *args_batch)
            d_cpu = {k: v.cpu() for k, v in d.items()}
            tiled_params_list.append(d_cpu)

    return collate(tiled_params_list, axis=axis)


def get_sep_catalog(images: torch.Tensor, *, slen: float, bp: float) -> FullCatalog:
    max_n_sources = 0
    all_sep_params = []
    for ii in range(images.shape[0]):
        im = images[ii, 0].numpy()
        bkg = sep.Background(im)
        catalog = sep.extract(
            im, err=bkg.globalrms, thresh=1.5, minarea=5, deblend_nthresh=32, deblend_cont=0.005
        )

        x1 = catalog["x"]
        y1 = catalog["y"]

        # need to ignore detected sources that are in the padding
        in_padding = (
            (x1 < bp - 0.5) | (x1 > bp + slen - 0.5) | (y1 < bp - 0.5) | (y1 > bp + slen - 0.5)
        )

        x = x1[np.logical_not(in_padding)]
        y = y1[np.logical_not(in_padding)]

        n = len(x)
        max_n_sources = max(n, max_n_sources)

        all_sep_params.append((n, x, y))

    n_sources = torch.zeros((images.shape[0],)).long()
    plocs = torch.zeros((images.shape[0], max_n_sources, 2))

    for jj in range(images.shape[0]):
        n, x, y = all_sep_params[jj]
        n_sources[jj] = n

        plocs[jj, :n, 0] = torch.from_numpy(y) - bp + 0.5
        plocs[jj, :n, 1] = torch.from_numpy(x) - bp + 0.5

    return FullCatalog(slen, slen, {"n_sources": n_sources, "plocs": plocs})
