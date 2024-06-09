import torch

from bliss.catalog import SourceType, TileCatalog
from bliss.encoder.unconstrained_dists import UnconstrainedBernoulli, UnconstrainedLogitNormal
from bliss.encoder.variational_dist import VariationalDist, VariationalDistSpec

# pylint: disable=duplicate-code


class GalaxyClusterVariationlDistSpec(VariationalDistSpec):
    def __init__(self, survey_bands, tile_slen):
        super().__init__(survey_bands, tile_slen)

        self.factor_specs["mem_prob"] = UnconstrainedBernoulli()
        self.factor_specs["galsim_hlr"] = UnconstrainedLogitNormal()
        self.factor_specs["galsim_g1"] = UnconstrainedLogitNormal()
        self.factor_specs["galsim_g2"] = UnconstrainedLogitNormal()

    def make_dist(self, x_cat):
        factors = self._parse_factors(x_cat)
        return GalaxyClusterVariationalDist(
            factors, survey_bands=self.survey_bands, tile_slen=self.tile_slen
        )


class GalaxyClusterVariationalDist(VariationalDist):
    def __init__(self, factors, survey_bands, tile_slen):
        super().__init__(factors, survey_bands, tile_slen)

        self.GALSIM_NAMES = ["hlr", "g1", "g2", "disk_frac"]

    def sample(self, use_mode=False) -> TileCatalog:
        """Sample the variational distribution.

        Args:
            use_mode: whether to use the mode of the distribution instead of random sampling

        Returns:
            TileCatalog: Sampled catalog
        """
        q = self.factors

        est_cat = {}

        locs = q["locs"].mode if use_mode else q["locs"].sample().squeeze(0)
        est_cat = {"locs": locs}

        # populate catalog with per-band (log) star fluxes
        sf_factors = [q[f"star_flux_{band}"] for band in self.survey_bands]
        sf_lst = [p.mode if use_mode else p.sample() for p in sf_factors]
        est_cat["star_fluxes"] = torch.stack(sf_lst, dim=3)

        # populate catalog with source type
        galaxy_bools = q["galaxy_prob"].mode if use_mode else q["galaxy_prob"].sample()
        galaxy_bools = galaxy_bools.unsqueeze(3)
        star_bools = 1 - galaxy_bools
        est_cat["source_type"] = SourceType.STAR * star_bools + SourceType.GALAXY * galaxy_bools

        # populate catalog with galaxy parameters
        gs_dists = [q[f"galsim_{name}"] for name in self.GALSIM_NAMES]
        gs_param_lst = [d.icdf(torch.tensor(0.5)) if use_mode else d.sample() for d in gs_dists]
        est_cat["galaxy_params"] = torch.stack(gs_param_lst, dim=3)

        # populate catalog with per-band galaxy fluxes
        gf_dists = [q[f"galaxy_flux_{band}"] for band in self.survey_bands]
        gf_lst = [d.icdf(torch.tensor(0.5)) if use_mode else d.sample() for d in gf_dists]
        est_cat["galaxy_fluxes"] = torch.stack(gf_lst, dim=3)

        est_cat["membership"] = q["mem_prob"].mode if use_mode else q["mem_prob"].sample()
        est_cat["membership"] = est_cat["membership"].unsqueeze(3)

        # we have to unsqueeze these tensors because a TileCatalog can store multiple
        # light sources per tile, but we sample only one source per tile
        for k, v in est_cat.items():
            est_cat[k] = v.unsqueeze(3)

        # n_sources is not unsqueezed because it is a single integer per tile regardless of
        # how many light sources are stored per tile

        est_cat["n_sources"] = q["on_prob"].mode if use_mode else q["on_prob"].sample()

        return TileCatalog(self.tile_slen, est_cat)

    def compute_nll(self, true_tile_cat: TileCatalog):
        q = self.factors

        # We just need counter loss for now
        # TODO:
        # redshift loss: can define ourselves,
        # or get it from redshift estimation (Declan's project)
        # fluxes loss: can be found in super().compute_nll() definition

        return -q["mem_prob"].log_prob(true_tile_cat["membership"].squeeze())
