from bliss.simulator.prior import CatalogPrior


class RedshiftPrior(CatalogPrior):
    """Prior distribution of objects in an astronomical image.

    Inherits from CatalogPrior, adding redshift. Temporarily,
    redshift is drawn from a tight uniform distribution Unif(0.99,1.01)
    for validation.
    """

    def __init__(
        self,
        redshift_min: float,
        redshift_max: float,
        *args,
        **kwargs,
    ):
        """Initializes CatalogPrior."""
        super().__init__(*args, **kwargs)
