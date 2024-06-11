from bliss.catalog import TileCatalog
from bliss.encoder.variational_dist import BernoulliFactor, LogitNormalFactor, VariationalDist


class GalaxyClusterVariationlDistSpec(VariationalDist):
    def __init__(self, survey_bands, tile_slen):
        super().__init__(survey_bands, tile_slen)

        self.factors["mem_prob"] = BernoulliFactor()
        self.factors["galsim_hlr"] = LogitNormalFactor()
        self.factors["galsim_g1"] = LogitNormalFactor()
        self.factors["galsim_g2"] = LogitNormalFactor()

    def make_dist(self, x_cat):
        factors = self._parse_factors(x_cat)
        return GalaxyClusterVariationalDist(factors, tile_slen=self.tile_slen)


class GalaxyClusterVariationalDist(VariationalDist):
    def __init__(self, factors, tile_slen):
        super().__init__(factors, tile_slen)

        self.GALSIM_NAMES = ["hlr", "g1", "g2", "disk_frac"]

    def sample(self, use_mode=False) -> TileCatalog:
        """Sample the variational distribution.

        Args:
            use_mode: whether to use the mode of the distribution instead of random sampling

        Returns:
            TileCatalog: Sampled catalog
        """
        est_cat = super().sample(use_mode=use_mode)

        q = self.factors
        est_cat["membership"] = q["mem_prob"].mode if use_mode else q["mem_prob"].sample()
        est_cat["membership"] = est_cat["membership"].unsqueeze(3).unsqueeze(3)

        return TileCatalog(self.tile_slen, est_cat)

    def compute_nll(self, true_tile_cat: TileCatalog):
        q = self.factors

        # We just need counter loss for now
        # TODO:
        # redshift loss: can define ourselves,
        # or get it from redshift estimation (Declan's project)
        # fluxes loss: can be found in super().compute_nll() definition

        return -q["mem_prob"].log_prob(true_tile_cat["membership"].squeeze())
