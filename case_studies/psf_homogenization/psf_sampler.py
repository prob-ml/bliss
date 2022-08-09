import galsim
import torch


class PsfSampler:
    def __init__(
        self,
        psf_rmin: float = 0.7,
        psf_rmax: float = 0.9,
    ) -> None:
        self.rmin = psf_rmin
        self.rmax = psf_rmax

    def sample(self) -> galsim.GSObject:
        # sample psf from galsim Gaussian distribution
        if self.rmin == self.rmax:
            fwhm = self.rmin
        elif self.rmin > self.rmax:
            raise ValueError("invalid argument!!!")
        else:
            fwhm = torch.distributions.uniform.Uniform(self.rmin, self.rmax).sample([1]).item()

        return galsim.Gaussian(fwhm=fwhm)
