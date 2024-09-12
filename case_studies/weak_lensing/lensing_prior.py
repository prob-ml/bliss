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

    def _apply_shear_and_convergence(self, shear, convergence, q, beta, a):
        complex_shear = shear[..., 0].unsqueeze(-1) + shear[..., 1].unsqueeze(-1) * 1j
        reduced_shear = complex_shear / (1.0 - convergence)
        magnification = 1 / (
            (1 - convergence) ** 2
            - shear[..., 0].unsqueeze(-1) ** 2
            - shear[..., 1].unsqueeze(-1) ** 2
        )

        e = (1 - q) / (1 + q)
        e1 = e * torch.cos(2 * beta)
        e2 = e * torch.sin(2 * beta)
        complex_e = e1 + e2 * 1j
        complex_e_lensed = (complex_e + reduced_shear) / (1.0 + reduced_shear.conj() * complex_e)
        galaxy_bulge_e_lensed = complex_e_lensed.absolute()
        q_lensed = (1 - galaxy_bulge_e_lensed) / (1 + galaxy_bulge_e_lensed)
        a_lensed = a * (q * magnification / q_lensed).sqrt()

        return q_lensed, a_lensed

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
        d["convergence"] = convergence

        d["galaxy_disk_q"], d["galaxy_a_d"] = self._apply_shear_and_convergence(
            d["shear"],
            d["convergence"],
            d["galaxy_disk_q"],
            d["galaxy_beta_radians"],
            d["galaxy_a_d"],
        )
        d["galaxy_bulge_q"], d["galaxy_a_b"] = self._apply_shear_and_convergence(
            d["shear"],
            d["convergence"],
            d["galaxy_bulge_q"],
            d["galaxy_beta_radians"],
            d["galaxy_a_b"],
        )

        return TileCatalog(d)
