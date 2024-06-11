from bliss.encoder.variational_dist import (
    BernoulliFactor,
    LogNormalFactor,
    TDBNFactor,
    VariationalDist,
)


class VariationalDistSpecExcludeGalaxyParams(VariationalDist):
    def __init__(self, survey_bands, tile_slen):
        super().__init__(survey_bands, tile_slen)

        self.factor_specs = {
            "n_sources": BernoulliFactor(),
            "locs": TDBNFactor(),
            "source_type": BernoulliFactor(),
            "star_fluxes": LogNormalFactor(dim=len(survey_bands)),
            "galaxy_fluxes": LogNormalFactor(dim=len(survey_bands)),
        }
