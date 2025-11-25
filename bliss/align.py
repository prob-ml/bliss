import numpy as np
from numpy import ndarray
from reproject import reproject_interp


def align(img: ndarray, wcs_list, ref_band: int, ref_depth: int = 0) -> ndarray:
    """Reproject images based on some reference WCS for pixel alignment.

    Args:
        img: 4D array of shape (coadd_depth, n_bands, h, w)
        wcs_list: list of lists, wcs_list[d][b] is the WCS for depth d, band b
        ref_band: reference band for alignment
        ref_depth: reference depth for alignment

    Returns:
        Reprojected images as 4D array of shape (coadd_depth, n_bands, h, w)
    """
    reproj_d = {}
    footprint_d = {}

    coadd_depth, n_bands, h, w = img.shape

    # align with (`ref_depth`, `ref_band`) WCS
    target_wcs = wcs_list[ref_depth][ref_band]
    for d in range(coadd_depth):
        for bnd in range(n_bands):
            inputs = (img[d, bnd], wcs_list[d][bnd])
            # the next line is the computational bottleneck
            reproj, footprint = reproject_interp(
                inputs, target_wcs, order="bicubic", shape_out=(h, w)
            )
            reproj_d[(d, bnd)] = reproj
            footprint_d[(d, bnd)] = footprint

    # use footprints to handle NaNs from reprojection
    fp_h, fp_w = footprint_d[(0, 0)].shape
    out_print = np.ones((fp_h, fp_w))
    for fp in footprint_d.values():
        out_print *= fp

    out_print = np.expand_dims(out_print, axis=(0, 1))
    reproj_out = np.zeros(img.shape)
    for d in range(coadd_depth):
        for bnd in range(n_bands):
            reproj_d[(d, bnd)] = np.multiply(reproj_d[(d, bnd)], out_print)
            cropped = reproj_d[(d, bnd)][0, 0, :h, :w]
            cropped[np.isnan(cropped)] = 0
            reproj_out[(d, bnd)] = cropped

    return np.float32(reproj_out)
