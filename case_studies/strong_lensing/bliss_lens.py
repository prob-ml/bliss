import sys

import galsim
import numpy as np
import torch


def sie_deflection(x_coord, y_coord, lens_params):
    """Calculate the deflection angles for a Singular Isothermal Ellipsoid (SIE) lens model.

    Args:
        x_coord (vector): Vector or image of x coordinates.
        y_coord (vector): Vector or image of y coordinates.
        lens_params (vector): Vector of parameters with 1 to 5 elements, defined as follows:
            par[0]: Lens strength, or 'Einstein radius'.
            par[1]: (optional) x-center.
            par[2]: (optional) y-center.
            par[3]: (optional) e1 ellipticity.
            par[4]: (optional) e2 ellipticity.

    Returns:
        tuple: (xg, yg) of gradients at the positions (x, y).

    Adopted from: Adam S. Bolton, U of Utah, 2009.
    """
    # Extract lens parameters
    params_np = lens_params.cpu().numpy()
    b_rad = params_np[0]
    center_x = params_np[1]
    center_y = params_np[2]
    e_1 = params_np[3]
    e_2 = params_np[4]
    q_val = (1 - (np.sqrt(e_1**2 + e_2**2))) / (1 + (np.sqrt(e_1**2 + e_2**2)))

    # Go into shifted coordinats of the potential:
    xsie = (x_coord - center_x) * np.cos(np.arctan(e_2 / e_1)) + (y_coord - center_y) * np.sin(
        np.arctan(e_2 / e_1)
    )

    ysie = (y_coord - center_y) * np.cos(np.arctan(e_2 / e_1)) - (x_coord - center_x) * np.sin(
        np.arctan(e_2 / e_1)
    )

    # Compute potential gradient in the transformed system:
    r_ell = np.sqrt(q_val * xsie**2 + ysie**2 / q_val)
    qfact = np.sqrt(1.0 / q_val - q_val)

    # (r_ell == 0) terms prevent divide-by-zero problems
    if qfact >= 0.001:
        xtg = (b_rad / qfact) * np.arctan(qfact * xsie / (r_ell + (r_ell == 0)))
        ytg = (b_rad / qfact) * np.arctanh(qfact * ysie / (r_ell + (r_ell == 0)))
    else:
        xtg = b_rad * xsie / (r_ell + (r_ell == 0))
        ytg = b_rad * ysie / (r_ell + (r_ell == 0))

    # Transform back to un-rotated system:
    return (
        xtg * np.cos(np.arctan(e_2 / e_1)) - ytg * np.sin(np.arctan(e_2 / e_1)),
        ytg * np.cos(np.arctan(e_2 / e_1)) + xtg * np.sin(np.arctan(e_2 / e_1)),
    )


def bilinear_interpolate_numpy(i_m, x_coord, y_coord):
    """Perform bilinear interpolation on a 2D numpy array.

    Args:
        i_m (numpy.ndarray): Input 2D array (image) to interpolate.
        x_coord (float): X-coordinate for interpolation.
        y_coord (float): Y-coordinate for interpolation.

    Returns:
        numpy.ndarray: Interpolated value at (x_coord, y_coord) using bilinear interpolation.

    Notes:
        - Uses numpy's floor, clip, and array indexing operations for efficient interpolation.
        - Assumes x_coord and y_coord are within the bounds of i_m.shape.
        - Bilinear interpolation computes a weighted average of the four nearest
          pixels around (x_coord, y_coord).
    """
    x_0 = np.floor(x_coord).astype(int)
    x_1 = x_0 + 1
    y_0 = np.floor(y_coord).astype(int)
    y_1 = y_0 + 1

    x_0 = np.clip(x_0, 0, i_m.shape[1] - 1)
    x_1 = np.clip(x_1, 0, i_m.shape[1] - 1)
    y_0 = np.clip(y_0, 0, i_m.shape[0] - 1)
    y_1 = np.clip(y_1, 0, i_m.shape[0] - 1)

    i_a = i_m[y_0, x_0]
    i_b = i_m[y_1, x_0]
    i_c = i_m[y_0, x_1]
    i_d = i_m[y_1, x_1]

    w_a = (x_1 - x_coord) * (y_1 - y_coord)
    w_b = (x_1 - x_coord) * (y_coord - y_0)
    w_c = (x_coord - x_0) * (y_1 - y_coord)
    w_d = (x_coord - x_0) * (y_coord - y_0)

    return (i_a.T * w_a).T + (i_b.T * w_b).T + (i_c.T * w_c).T + (i_d.T * w_d).T


def lens_galsim(unlensed_image, lens_params):
    """Apply gravitational lensing to input image using Singular Isothermal Ellipsoid (SIE) model.

    Args:
        unlensed_image (numpy.ndarray): Input 2D array representing the unlensed image.
        lens_params (numpy.ndarray): Parameters for the SIE lens model.
            lens_params should contain:
            - lens strength (Einstein radius)
            - x-center (optional)
            - y-center (optional)
            - e1 ellipticity (optional)
            - e2 ellipticity (optional)

    Returns:
        numpy.ndarray: Lensed image after applying gravitational lensing.
            Has the same dtype as the input `unlensed_image`.

    Notes:
        - Uses numpy operations to generate coordinates and perform deflection.
        - Calls `sie_deflection` to compute deflection angles for gravitational lensing.
        - Uses `bilinear_interpolate_numpy` for bilinear interpolation of pixel values.
        - Assumes the input image `unlensed_image` is centered at the origin.
    """
    n_x, n_y = unlensed_image.shape
    x_range = [-n_x // 2, n_x // 2]
    y_range = [-n_y // 2, n_y // 2]
    x_coord = (x_range[1] - x_range[0]) * np.outer(np.ones(n_y), np.arange(n_x)) / float(
        n_x - 1
    ) + x_range[0]
    y_coord = (y_range[1] - y_range[0]) * np.outer(np.arange(n_y), np.ones(n_x)) / float(
        n_y - 1
    ) + y_range[0]

    (x_g, y_g) = sie_deflection(x_coord, y_coord, lens_params)
    lensed_image = bilinear_interpolate_numpy(
        unlensed_image, (x_coord - x_g) + n_x // 2, (y_coord - y_g) + n_y // 2
    )
    return lensed_image.astype(unlensed_image.dtype)


IMAGE_PATH = sys.argv[1]
image = galsim.fits.read(IMAGE_PATH)
image = image.array

theta_e = np.random.uniform(10, 20)  # Einstein radius
x_center = np.random.uniform(-10, 10)  # Center x-coordinate
y_center = np.random.uniform(-10, 10)  # Center y-coordinate
e1 = np.random.uniform(0.1, 0.7)  # Ellipticity component e1
e2 = np.random.uniform(0.1, 0.7)  # Ellipticity component e2

# Initialize lens_params as a PyTorch tensor
params = torch.tensor([theta_e, x_center, y_center, e1, e2])  # pylint: disable=E1101

lensed_img = lens_galsim(image, params)


OUTPUT_DIR = sys.argv[2]
lensed_img = galsim.ImageF(lensed_img)
lensed_img.write(f"{OUTPUT_DIR}/galsim.fits")

params = [theta_e, x_center, y_center, e1, e2]
print(" ".join(map(str, params)))  # noqa: WPS421
