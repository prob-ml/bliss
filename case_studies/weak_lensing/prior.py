from bliss.simulator.prior import CatalogPrior

class LensingPrior(CatalogPrior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)