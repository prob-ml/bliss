import torch
from einops import rearrange
from torch.distributions import Uniform

from bliss.catalog import TileCatalog
from bliss.simulator.prior import CatalogPrior


class LensingPrior(CatalogPrior):
    def __init__(
        self,
        *args,
        shear_min,
        shear_max,
        convergence_min,
        convergence_max,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shear_min = shear_min
        self.shear_max = shear_max
        self.convergence_min = convergence_min
        self.convergence_max = convergence_max

    def _sample_shear_and_convergence(self):
        latent_dims_shear = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 2)
        shear = rearrange(
            Uniform(self.shear_min, self.shear_max).sample([self.batch_size, 2]), "b d -> b 1 1 1 d"
        ) * torch.ones(latent_dims_shear)

        latent_dims_convergence = (
            self.batch_size,
            self.n_tiles_h,
            self.n_tiles_w,
            self.max_sources,
            1,
        )
        convergence = rearrange(
            Uniform(self.convergence_min, self.convergence_max).sample([self.batch_size, 1]),
            "b d -> b 1 1 1 d",
        ) * torch.ones(latent_dims_convergence)

        return shear, convergence

    def sample(self) -> TileCatalog:
        """Samples latent variables from the prior of an astronomical image.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first three dimensions of each tensor are
            `(batch_size, self.n_tiles_h, self.n_tiles_w)`.
            The remaining dimensions are variable-specific.
        """

        d = super().sample()

        shear, convergence = self._sample_shear_and_convergence()

        d["shear_per_galaxy"] = shear
        d["shear_1_per_galaxy"] = shear[..., 0].unsqueeze(-1)
        d["shear_2_per_galaxy"] = shear[..., 1].unsqueeze(-1)
        d["shear"] = d["shear_per_galaxy"].mean(-2)
        d["shear_1"] = d["shear_1_per_galaxy"].mean(-2)
        d["shear_2"] = d["shear_2_per_galaxy"].mean(-2)

        d["convergence_per_galaxy"] = convergence
        d["convergence"] = d["convergence_per_galaxy"].mean(-2)

        return TileCatalog(d)
