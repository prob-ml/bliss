"""Utility functions for DC2 data processing.

These are local copies of utilities from bliss core modules to make the
weak lensing DC2 case study fully standalone.
"""

import collections

import numpy as np
import torch
from astropy.io.fits import Header
from astropy.wcs import WCS


def wcs_from_wcs_header_str(wcs_header_str: str):
    """Parse WCS from FITS header string."""
    return WCS(Header.fromstring(wcs_header_str))


def map_nested_dicts(cur_dict, func):
    """Recursively apply a function to all values in a nested dict."""
    if isinstance(cur_dict, collections.abc.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in cur_dict.items()}
    return func(cur_dict)


def unpack_dict(ori_dict):
    """Convert dict of lists to list of dicts."""
    return [dict(zip(ori_dict, v, strict=True)) for v in zip(*ori_dict.values(), strict=True)]


def split_tensor(ori_tensor, split_size, split_first_dim, split_second_dim):
    """Split a tensor along two dimensions into a list of sub-tensors."""
    tensor_splits = torch.stack(ori_tensor.split(split_size, dim=split_first_dim))
    tensor_splits = torch.stack(tensor_splits.split(split_size, dim=split_second_dim + 1), dim=1)
    return [sub_tensor.squeeze(0) for sub_tensor in tensor_splits.flatten(0, 1).split(1, dim=0)]


def plocs_from_ra_dec(ras, decs, wcs):
    """Convert RA/DEC coordinates to BLISS pixel coordinates.

    BLISS pixel coordinates have (0, 0) as the lower-left corner, whereas standard pixel
    coordinates begin at (-0.5, -0.5). BLISS pixel coordinates are in row-column order,
    whereas standard pixel coordinates are in column-row order.

    Args:
        ras: Tensor of RA coordinates in degrees.
        decs: Tensor of DEC coordinates in degrees.
        wcs: WCS object to use for transformation.

    Returns:
        Tensor containing the locations in pixel coordinates (row, col).
    """
    ras = ras.numpy().squeeze()
    decs = decs.numpy().squeeze()

    pt, pr = wcs.all_world2pix(ras, decs, 0)  # convert to pixel coordinates
    pt = torch.tensor(pt) + 0.5  # For consistency with BLISS
    pr = torch.tensor(pr) + 0.5
    return torch.stack((pr, pt), dim=-1)


def get_bands_flux_and_psf(bands, catalog, median=True):
    """Extract flux and PSF parameters for specified bands from catalog.

    Args:
        bands: Tuple of band names (e.g., ('u', 'g', 'r', 'i', 'z', 'y'))
        catalog: Pandas DataFrame with catalog data
        median: If True, return median PSF params; if False, return per-object params

    Returns:
        Tuple of (fluxes, psf_params) tensors
    """
    flux_list = []
    psf_params_list = []
    for b in bands:
        flux_list.append(torch.from_numpy((catalog[f"flux_{b}"]).values))
        psf_params_name = ["IxxPSF_pixel_", "IyyPSF_pixel_", "IxyPSF_pixel_", "psf_fwhm_"]
        psf_params_cur_band = []
        for i in psf_params_name:
            if median:
                median_psf = np.nanmedian((catalog[f"{i}{b}"]).values).astype(np.float32)
                psf_params_cur_band.append(median_psf)
            else:
                psf_params_cur_band.append(catalog[f"{i}{b}"].values.astype(np.float32))
        psf_params_list.append(torch.tensor(np.array(psf_params_cur_band)))

    return torch.stack(flux_list).t(), torch.stack(psf_params_list).unsqueeze(0)
