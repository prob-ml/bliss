# %%
import numpy as np
from astropy.table import Table
from astropy.wcs.wcs import WCS
from matplotlib import pyplot as plt

from bliss.datasets.galsim_galaxies import load_psf_from_file
from bliss.reporting import get_flux_coadd, get_hlr_coadd

# %%
# scatter plot of miscclassification probs
prob_galaxy = np.zeros((10,))
misclass = np.zeros((10,))
true_mags = np.zeros((10,))
probs_correct = prob_galaxy[~misclass]
probs_misclass = prob_galaxy[misclass]

plt.scatter(true_mags[~misclass], probs_correct, marker="x", c="b")
plt.scatter(true_mags[misclass], probs_misclass, marker="x", c="r")
plt.axhline(0.5, linestyle="--")
plt.axhline(0.1, linestyle="--")
plt.axhline(0.9, linestyle="--")

uncertain = (prob_galaxy[misclass] > 0.2) & (prob_galaxy[misclass] < 0.8)
r_uncertain = sum(uncertain) / len(prob_galaxy[misclass])
print(
    f"ratio misclass with probability between 10%-90%: {r_uncertain:.3f}",
)


# %%
def add_extra_coadd_info(coadd_cat_file: str, psf_image_file: str, pixel_scale: float, wcs: WCS):
    """Add additional useful information to coadd catalog."""
    coadd_cat = Table.read(coadd_cat_file)

    psf = load_psf_from_file(psf_image_file, pixel_scale)
    x, y = wcs.all_world2pix(coadd_cat["ra"], coadd_cat["dec"], 0)
    galaxy_bool = ~coadd_cat["probpsf"].data.astype(bool)
    flux, mag = get_flux_coadd(coadd_cat)
    hlr = get_hlr_coadd(coadd_cat, psf)

    coadd_cat["x"] = x
    coadd_cat["y"] = y
    coadd_cat["galaxy_bool"] = galaxy_bool
    coadd_cat["flux"] = flux
    coadd_cat["mag"] = mag
    coadd_cat["hlr"] = hlr
    coadd_cat.replace_column("is_saturated", coadd_cat["is_saturated"].data.astype(bool))
    coadd_cat.write(coadd_cat_file, overwrite=True)  # overwrite with additional info.
