import galsim
import numpy as np

from bliss.simulator.decoder import Decoder


class LensingDecoder(Decoder):
    def render_galaxy(self, psf, band, source_params):
        """Render a galaxy with given params and PSF.

        Args:
            psf (List): a list of PSFs for each band
            band (int): band
            source_params (Tensor): Tensor containing the parameters for a particular source
                (see prior.py for details about these parameters)

        Returns:
            GSObject: a galsim representation of the rendered galaxy convolved with the PSF
        """
        disk_flux = source_params["galaxy_fluxes"][band] * source_params["galaxy_disk_frac"]
        bulge_frac = 1 - source_params["galaxy_disk_frac"]
        bulge_flux = source_params["galaxy_fluxes"][band] * bulge_frac
        beta = source_params["galaxy_beta_radians"] * galsim.radians

        components = []
        if disk_flux > 0:
            b_d = source_params["galaxy_a_d"] * source_params["galaxy_disk_q"]
            disk_hlr_arcsecs = np.sqrt(source_params["galaxy_a_d"] * b_d)
            disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs)
            sheared_disk = disk.shear(q=source_params["galaxy_disk_q"].item(), beta=beta)
            components.append(sheared_disk)
        if bulge_flux > 0:
            b_b = source_params["galaxy_a_b"] * source_params["galaxy_bulge_q"]
            bulge_hlr_arcsecs = np.sqrt(source_params["galaxy_a_b"] * b_b)
            bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs)
            sheared_bulge = bulge.shear(q=source_params["galaxy_bulge_q"].item(), beta=beta)
            components.append(sheared_bulge)
        galaxy = galsim.Add(components)

        shear = source_params["shear"]
        shear1, shear2 = shear
        convergence = source_params["convergence"]

        reduced_shear1 = shear1 / (1 - convergence)
        reduced_shear2 = shear2 / (1 - convergence)
        magnification = 1 / ((1 - convergence) ** 2 - shear1**2 - shear2**2)

        lensed_galaxy = galaxy.lens(
            g1=reduced_shear1.item(), g2=reduced_shear2.item(), mu=magnification.item()
        )

        return galsim.Convolution(lensed_galaxy, psf[band])
