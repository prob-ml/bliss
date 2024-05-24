from bliss.catalog import TileCatalog
from bliss.encoder.variational_dist import VariationalDistSpec, VariationalDist
from bliss.encoder.unconstrained_dists import UnconstrainedBernoulli




class GalaxyClusterVariationlDistSpec(VariationalDistSpec):
    def __init__(self, survey_bands, tile_slen):
        super().__init__(survey_bands, tile_slen)

        self.factor_specs["mem_prob"] = UnconstrainedBernoulli()

    def make_dist(self, x_cat):
        factors = self._parse_factors(x_cat)
        return GalaxyClusterVariationalDist(factors, survey_bands=self.survey_bands, tile_slen=self.tile_slen)
    



class GalaxyClusterVariationalDist(VariationalDist):
    def __init__(self, factors, survey_bands, tile_slen):
        super().__init__(factors, survey_bands, tile_slen)

    def sample(self, use_mode=False) -> TileCatalog:
        """Sample the variational distribution.

        Args:
            use_mode: whether to use the mode of the distribution instead of random sampling

        Returns:
            TileCatalog: Sampled catalog
        """
        q = self.factors

        est_cat = {}
        est_cat["membership"] = q["mem_prob"].mode if use_mode else q["mem_prob"].sample()

        return TileCatalog(self.tile_slen, est_cat)

    def compute_nll(self, true_tile_cat: TileCatalog):
        q = self.factors

        # We just need counter loss for now
        counter_loss = -q["mem_prob"].log_prob(true_tile_cat["membership"])
        loss = counter_loss

        # TODO:
        ## redshift loss: can define ourselves, or get it from redshift estimation (Declan's project)
        ## fluxes loss: can be found in super().compute_nll() definition

        return loss