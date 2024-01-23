from lensing_catalog import LensingTileCatalog

from bliss.encoder.unconstrained_dists import UnconstrainedLogitNormal, UnconstrainedTDBN
from bliss.encoder.variational_dist import VariationalDist, VariationalDistSpec


class LensingVariationalDistSpec(VariationalDistSpec):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.factor_specs = {
            "shear": UnconstrainedTDBN(),
            "convergence": UnconstrainedLogitNormal(),
        }

    def make_dist(self, x_cat):
        # override this method to instantiate a subclass of VariationalGrid, e.g.,
        # one with additional distribution parameter groups
        factors = self._parse_factors(x_cat)
        return LensingVariationalDist(factors, self.survey_bands, self.tile_slen)


class LensingVariationalDist(VariationalDist):
    def sample(self, use_mode=False) -> LensingTileCatalog:
        """Sample the variational distribution.

        Args:
            use_mode: whether to use the mode of the distribution instead of random sampling

        Returns:
            LensingTileCatalog: Sampled catalog
        """
        q = self.factors

        est_cat = {}
        est_cat["shear"] = q["shear"].mode if use_mode else q["shear"].sample()
        est_cat["convergence"] = q["convergence"].mode if use_mode else q["convergence"].sample()

        return LensingTileCatalog(self.tile_slen, est_cat)

    def compute_nll(self, true_tile_cat: LensingTileCatalog):
        q = self.factors

        true_shear = true_tile_cat.data["shear"].squeeze()
        true_convergence = true_tile_cat.data["convergence"].squeeze()
        lensing_loss = -q["shear"].log_prob(true_shear)
        lensing_loss -= q["convergence"].log_prob(true_convergence)

        return lensing_loss
