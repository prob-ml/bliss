import numpy as np
from reproject import reproject_interp


def crop_to_mult16(x):
    """Crop the image dimensions to a multiple of 16."""
    # note: by cropping the top-right, we preserve the mapping between pixel coordinates
    # and the original WCS coordinates
    height = x.shape[1] - (x.shape[1] % 16)
    width = x.shape[2] - (x.shape[2] % 16)
    return x[:, :height, :width]


def align(img, wcs_list, ref_band, ref_depth=0):
    """Reproject images based on some reference WCS for pixel alignment."""
    reproj_d = {}
    footprint_d = {}

    # coerce to 4D with coadd_depth trivially 1
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    if not isinstance(wcs_list[0], list):
        wcs_list = [wcs_list]
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

    if reproj_out.shape[0] == 1:
        reproj_out = reproj_out.squeeze(axis=0)
    return reproj_out
