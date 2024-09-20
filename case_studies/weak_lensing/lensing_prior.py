import torch

from bliss.catalog import TileCatalog
from bliss.simulator.prior import CatalogPrior


class LensingPrior(CatalogPrior):
    def __init__(
        self,
        *args,
        constant_shear,
        constant_convergence,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.constant_shear = constant_shear
        self.constant_convergence = constant_convergence

    def _sample_shear_and_convergence(self):
        shear = self.constant_shear * torch.ones(
            (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 2)
        )
        convergence = self.constant_convergence * torch.ones(
            (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        )

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

        d["shear"] = shear
        d["shear_1"] = shear[..., 0].unsqueeze(-1)
        d["shear_2"] = shear[..., 1].unsqueeze(-1)
        d["shear_avg"] = shear.mean(-2).unsqueeze(-2)
        d["shear_1_avg"] = d["shear_1"].mean(-2).unsqueeze(-2)
        d["shear_2_avg"] = d["shear_2"].mean(-2).unsqueeze(-2)

        d["convergence"] = convergence
        d["convergence_avg"] = convergence.mean(-2).unsqueeze(-2)

        return TileCatalog(d)
