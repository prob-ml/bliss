from bliss.simulator.prior import CatalogPrior


class LensingPrior(CatalogPrior):
    def __init__(
        self,
        *args,
        shear_horizontal: float = 0.0,
        shear_diagonal: float = 0.0,
        convergence: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shear_horizontal = shear_horizontal
        self.shear_diagonal = shear_diagonal
        self.convergence = convergence
