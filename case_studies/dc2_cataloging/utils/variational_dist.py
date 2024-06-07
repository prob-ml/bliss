from bliss.encoder.unconstrained_dists import (
    UnconstrainedBernoulli,
    UnconstrainedLogNormal,
    UnconstrainedTDBN,
)
from bliss.encoder.variational_dist import VariationalDistSpec


class VariationalDistSpecExcludeGalaxyParams(VariationalDistSpec):
    def __init__(self, survey_bands, tile_slen):
        super().__init__(survey_bands, tile_slen)

        self.factor_specs = {
            "on_prob": UnconstrainedBernoulli(),
            "loc": UnconstrainedTDBN(),
            "galaxy_prob": UnconstrainedBernoulli(),
        }
        for band in survey_bands:
            self.factor_specs[f"star_flux_{band}"] = UnconstrainedLogNormal()
        for band in survey_bands:
            self.factor_specs[f"galaxy_flux_{band}"] = UnconstrainedLogNormal()
