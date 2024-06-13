from bliss.catalog import TileCatalog
from bliss.encoder.unconstrained_dists import UnconstrainedNormal
from bliss.encoder.variational_dist import VariationalDist, VariationalDistSpec


class RedshiftVariationalDistSpec(VariationalDistSpec):
    def __init__(self, survey_bands, tile_slen):
        super().__init__(survey_bands, tile_slen)
        self.factor_specs["redshifts"] = UnconstrainedNormal()

    def make_dist(self, x_cat):
        factors = self._parse_factors(x_cat)
        return RedshiftVariationalDist(factors, self.tile_slen)


class RedshiftVariationalDist(VariationalDist):
    def sample(self, use_mode=False) -> TileCatalog:
        """Sample the variational distribution.

        Args:
            use_mode: whether to use the mode of the distribution instead of random sampling

        Returns:
            TileCatalog: Sampled catalog
        """
        est_cat = super().sample(use_mode=use_mode)

        q = self.factors
        est_cat["redshifts"] = q["redshifts"].mode if use_mode else q["redshifts"].sample()
        # chnage from 3d to 5d
        est_cat["redshifts"] = est_cat["redshifts"].unsqueeze(-1).unsqueeze(-1)

        return TileCatalog(self.tile_slen, est_cat)

    def compute_nll(self, true_tile_cat: TileCatalog):
        loss = super().compute_nll(true_tile_cat)
        q = self.factors

        true_redshifts = true_tile_cat["redshifts"].squeeze()
        redshift_loss = -q["redshifts"].log_prob(true_redshifts).squeeze(-1)
        loss += redshift_loss

        return loss
