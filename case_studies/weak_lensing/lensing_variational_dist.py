import torch

from bliss.catalog import SourceType, TileCatalog
from bliss.encoder.unconstrained_dists import ShearUnconstrainedNormal, UnconstrainedNormal
from bliss.encoder.variational_dist import VariationalDist, VariationalDistSpec


class LensingVariationalDistSpec(VariationalDistSpec):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.factor_specs["shear"] = ShearUnconstrainedNormal()
        self.factor_specs["convergence"] = UnconstrainedNormal()

    def make_dist(self, x_cat):
        # override this method to instantiate a subclass of VariationalGrid, e.g.,
        # one with additional distribution parameter groups
        factors = self._parse_factors(x_cat)
        return LensingVariationalDist(factors, self.survey_bands, self.tile_slen)


class LensingVariationalDist(VariationalDist):
    def sample(self, use_mode=False) -> TileCatalog:
        """Sample the variational distribution.

        Args:
            use_mode: whether to use the mode of the distribution instead of random sampling

        Returns:
            TileCatalog: Sampled catalog
        """
        q = self.factors

        locs = q["loc"].mode if use_mode else q["loc"].sample().squeeze(0)
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

        # populate catalog with shear and convergence
        est_cat["shear"] = q["shear"].mode if use_mode else q["shear"].sample().squeeze(0)
        est_cat["convergence"] = (
            q["convergence"].mode.unsqueeze(3)
            if use_mode
            else q["convergence"].sample().unsqueeze(3)
        )

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

        # Squeezing here won't work if max_sources > 1, so this might need to be rewritten
        true_shear = true_tile_cat.data["shear"].squeeze(3)
        true_convergence = true_tile_cat.data["convergence"].squeeze((3, 4))
        lensing_loss = -q["shear"].log_prob(true_shear)
        lensing_loss -= q["convergence"].log_prob(true_convergence)

        return lensing_loss
