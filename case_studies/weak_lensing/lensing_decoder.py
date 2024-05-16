import galsim

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
        galaxy = super().render_galaxy(psf, band, source_params)

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
