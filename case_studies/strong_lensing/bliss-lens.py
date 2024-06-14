import numpy as np
import galsim
import torch
import sys


def sie_deflection(x, y, lens_params):
        """
        args:
        x, y: vectors or images of coordinates;
        par: vector of parameters with 1 to 5 elements, defined as follows:
            par[0]: lens strength, or 'Einstein radius'
            par[1]: (optional) x-center
            par[2]: (optional) y-center
            par[3]: (optional) e1 ellipticity
            par[4]: (optional) e2 ellipticity
        RETURNS: tuple (xg, yg) of gradients at the positions (x, y)
        Adopted from: Adam S. Bolton, U of Utah, 2009
        """
        b, center_x, center_y, e1, e2 = lens_params.cpu().numpy()
        ell = np.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - ell) / (1 + ell)
        phirad = np.arctan(e2 / e1)

        # Go into shifted coordinats of the potential:
        xsie = (x-center_x) * np.cos(phirad) + (y-center_y) * np.sin(phirad)
        ysie = (y-center_y) * np.cos(phirad) - (x-center_x) * np.sin(phirad)

        # Compute potential gradient in the transformed system:
        r_ell = np.sqrt(q * xsie ** 2 + ysie ** 2 / q)
        qfact = np.sqrt(1./q - q)

        # (r_ell == 0) terms prevent divide-by-zero problems
        eps = 0.001
        if (qfact >= eps):
            xtg = (b/qfact) * np.arctan(qfact * xsie / (r_ell + (r_ell == 0)))
            ytg = (b/qfact) * np.arctanh(qfact * ysie / (r_ell + (r_ell == 0)))
        else:
            xtg = b * xsie / (r_ell + (r_ell == 0))
            ytg = b * ysie / (r_ell + (r_ell == 0))

        # Transform back to un-rotated system:
        xg = xtg * np.cos(phirad) - ytg * np.sin(phirad)
        yg = ytg * np.cos(phirad) + xtg * np.sin(phirad)
        return (xg, yg)

def bilinear_interpolate_numpy(im, x, y):
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1]-1)
        x1 = np.clip(x1, 0, im.shape[1]-1)
        y0 = np.clip(y0, 0, im.shape[0]-1)
        y1 = np.clip(y1, 0, im.shape[0]-1)

        Ia = im[ y0, x0 ]
        Ib = im[ y1, x0 ]
        Ic = im[ y0, x1 ]
        Id = im[ y1, x1 ]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T

def lens_galsim(unlensed_image, lens_params):
        nx, ny = unlensed_image.shape
        x_range = [-nx // 2, nx // 2]
        y_range = [-ny // 2, ny // 2]
        x = (x_range[1] - x_range[0]) * np.outer(np.ones(ny), np.arange(nx)) / float(nx-1) + x_range[0]
        y = (y_range[1] - y_range[0]) * np.outer(np.arange(ny), np.ones(nx)) / float(ny-1) + y_range[0]

        (xg, yg) = sie_deflection(x, y, lens_params)
        lensed_image = bilinear_interpolate_numpy(unlensed_image, (x-xg) + nx // 2, (y-yg) + ny // 2)
        return lensed_image.astype(unlensed_image.dtype)




image_path = sys.argv[1]
image = galsim.fits.read(image_path)
image = image.array

theta_E = np.random.uniform(10, 20)  # Einstein radius
center_x = np.random.uniform(-10, 10)  # Center x-coordinate
center_y = np.random.uniform(-10, 10)  # Center y-coordinate
e1 = np.random.uniform(0.1, 0.7)  # Ellipticity component e1
e2 = np.random.uniform(0.1, 0.7) # Ellipticity component e2

# Initialize lens_params as a PyTorch tensor
lens_params = torch.tensor([theta_E, center_x, center_y, e1, e2])

lensed_img = lens_galsim(image, lens_params)
# print(lensed_img.shape)
# print(image.shape)



output_dir = sys.argv[2]
lensed_image_path = sys.argv[3]
lensed_image = galsim.ImageF(lensed_img)
lensed_image.write(output_dir + '/lensed_image.fits')


from astropy.io import fits
import matplotlib.pyplot as plt

image_file = output_dir + "/lensed_image.fits"
image_data = fits.getdata(image_file)


c = plt.imshow(image_data)
plt.colorbar(c)
plt.savefig(lensed_image_path)


