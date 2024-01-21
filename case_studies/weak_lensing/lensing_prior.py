from bliss.simulator.prior import CatalogPrior


class LensingPrior(CatalogPrior):
    def __init__(
        self,
        *args,
        shear_min: float = -0.5,
        shear_max: float = 0.5,
        convergence_a: float = 1,
        convergence_b: float = 100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shear_min = shear_min
        self.shear_max = shear_max
        self.convergence_a = convergence_a
        self.convergence_b = convergence_b
