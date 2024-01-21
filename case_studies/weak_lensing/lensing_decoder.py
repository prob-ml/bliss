import galsim
import numpy as np

from bliss.simulator.decoder import ImageDecoder


class LensingDecoder(ImageDecoder):
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
        galaxy_fluxes = source_params["galaxy_fluxes"]
        galaxy_params = source_params["galaxy_params"]

        total_flux = galaxy_fluxes[band]
        disk_frac, beta_radians, disk_q, a_d, bulge_q, a_b = galaxy_params

        disk_flux = total_flux * disk_frac
        bulge_frac = 1 - disk_frac
        bulge_flux = total_flux * bulge_frac

        components = []
        if disk_flux > 0:
            b_d = a_d * disk_q
            disk_hlr_arcsecs = np.sqrt(a_d * b_d)
            disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs)
            sheared_disk = disk.shear(q=disk_q, beta=beta_radians * galsim.radians)
            components.append(sheared_disk)
        if bulge_flux > 0:
            b_b = bulge_q * a_b
            bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
            bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs)
            sheared_bulge = bulge.shear(q=bulge_q, beta=beta_radians * galsim.radians)
            components.append(sheared_bulge)
        galaxy = galsim.Add(components)
        return galsim.Convolution(galaxy, psf[band])
