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
            "n_sources": UnconstrainedBernoulli(),
            "locs": UnconstrainedTDBN(),
            "source_type": UnconstrainedBernoulli(),
            "star_fluxes": UnconstrainedLogNormal(dim=len(survey_bands)),
            "galaxy_fluxes": UnconstrainedLogNormal(dim=len(survey_bands)),
        }
