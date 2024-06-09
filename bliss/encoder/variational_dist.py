from typing import List

import torch
from einops import rearrange

from bliss.catalog import SourceType, TileCatalog
from bliss.encoder.unconstrained_dists import (
    UnconstrainedBernoulli,
    UnconstrainedLogitNormal,
    UnconstrainedLogNormal,
    UnconstrainedTDBN,
)


class VariationalDistSpec(torch.nn.Module):
    def __init__(self, survey_bands, tile_slen):
        super().__init__()

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen

        # overriding this dict in subclass enables you to exclude loss
        self.factor_specs = {
            "on_prob": UnconstrainedBernoulli(),
            "loc": UnconstrainedTDBN(low_clamp=-5),
            "galaxy_prob": UnconstrainedBernoulli(),
            # galsim parameters
            "galsim_disk_frac": UnconstrainedLogitNormal(),
            "galsim_beta_radians": UnconstrainedLogitNormal(high=torch.pi),
            "galsim_disk_q": UnconstrainedLogitNormal(),
            "galsim_a_d": UnconstrainedLogNormal(),
            "galsim_bulge_q": UnconstrainedLogitNormal(),
            "galsim_a_b": UnconstrainedLogNormal(),
        }
        for band in survey_bands:
            self.factor_specs[f"star_flux_{band}"] = UnconstrainedLogNormal()
        for band in survey_bands:
            self.factor_specs[f"galaxy_flux_{band}"] = UnconstrainedLogNormal()

    @property
    def n_params_per_source(self):
        return sum(param.dim for param in self.factor_specs.values())

    def _parse_factors(self, x_cat):
        split_sizes = [v.dim for v in self.factor_specs.values()]
        dist_params_split = torch.split(x_cat, split_sizes, 3)
        names = self.factor_specs.keys()
        factors = dict(zip(names, dist_params_split))

        for k, v in factors.items():
            factors[k] = self.factor_specs[k].get_dist(v)

        return factors

    def make_dist(self, x_cat):
        # override this method to instantiate a subclass of VariationalGrid, e.g.,
        # one with additional distribution parameter groups
        factors = self._parse_factors(x_cat)
        return VariationalDist(
            factors,
            self.survey_bands,
            self.tile_slen,
        )


class VariationalDist(torch.nn.Module):
    GALSIM_NAMES = ["disk_frac", "beta_radians", "disk_q", "a_d", "bulge_q", "a_b"]

    def __init__(self, factors, survey_bands, tile_slen):
        super().__init__()

        self.factors = factors
        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.factors_name_set = set(self.factors.keys())

        self.loc_name_lst = ["loc"]
        self.star_flux_name_lst = [f"star_flux_{band}" for band in self.survey_bands]
        self.source_type_name_lst = ["galaxy_prob"]
        self.galaxy_params_name_lst = [f"galsim_{name}" for name in self.GALSIM_NAMES]
        self.galaxy_flux_name_lst = [f"galaxy_flux_{band}" for band in self.survey_bands]
        self.n_sources_name_lst = ["on_prob"]

        self.loc_available = self._factors_are_available(self.loc_name_lst)
        self.star_flux_available = self._factors_are_available(self.star_flux_name_lst)
        self.source_type_available = self._factors_are_available(self.source_type_name_lst)
        self.galaxy_params_available = self._factors_are_available(self.galaxy_params_name_lst)
        self.galaxy_flux_available = self._factors_are_available(self.galaxy_flux_name_lst)
        self.n_sources_available = self._factors_are_available(self.n_sources_name_lst)

        assert self.loc_available, "factors should have loc"
        assert self.n_sources_available, "factors should have on_prob"

        two_type_flux = self.star_flux_available == self.galaxy_flux_available
        error_msg = "you only use one type (star/galaxy) of flux error, which is not permitted"
        assert two_type_flux, error_msg

    def _factors_are_available(self, factors_name_list: List[str]) -> bool:
        return set(factors_name_list).issubset(self.factors_name_set)

    def sample(self, use_mode=False) -> TileCatalog:
        """Sample the variational distribution.

        Args:
            use_mode: whether to use the mode of the distribution instead of random sampling

        Returns:
            TileCatalog: Sampled catalog
        """
        q = self.factors

        est_cat = {}

        if self.loc_available:
            locs = q["loc"].mode if use_mode else q["loc"].sample().squeeze(0)
            est_cat["locs"] = locs

        # populate catalog with per-band (log) star fluxes
        if self.star_flux_available:
            sf_factors = [q[factor] for factor in self.star_flux_name_lst]
            sf_lst = [p.mode if use_mode else p.sample() for p in sf_factors]
            est_cat["star_fluxes"] = torch.stack(sf_lst, dim=3)

        # populate catalog with source type
        if self.source_type_available:
            galaxy_bools = q["galaxy_prob"].mode if use_mode else q["galaxy_prob"].sample()
            galaxy_bools = galaxy_bools.unsqueeze(3)
            star_bools = 1 - galaxy_bools
            est_cat["source_type"] = SourceType.STAR * star_bools + SourceType.GALAXY * galaxy_bools

        # populate catalog with galaxy parameters
        if self.galaxy_params_available:
            gs_dists = [q[factor] for factor in self.galaxy_params_name_lst]
            gs_param_lst = [d.icdf(torch.tensor(0.5)) if use_mode else d.sample() for d in gs_dists]
            est_cat["galaxy_params"] = torch.stack(gs_param_lst, dim=3)

        # populate catalog with per-band galaxy fluxes
        if self.galaxy_flux_available:
            gf_dists = [q[factor] for factor in self.galaxy_flux_name_lst]
            gf_lst = [d.icdf(torch.tensor(0.5)) if use_mode else d.sample() for d in gf_dists]
            est_cat["galaxy_fluxes"] = torch.stack(gf_lst, dim=3)

        # we have to unsqueeze these tensors because a TileCatalog can store multiple
        # light sources per tile, but we sample only one source per tile
        for k, v in est_cat.items():
            est_cat[k] = v.unsqueeze(3)

        # n_sources is not unsqueezed because it is a single integer per tile regardless of
        # how many light sources are stored per tile
        if self.n_sources_available:
            est_cat["n_sources"] = q["on_prob"].mode if use_mode else q["on_prob"].sample()

        return TileCatalog(self.tile_slen, est_cat)

    def compute_nll(self, true_tile_cat: TileCatalog):
        q = self.factors

        # counter loss
        if self.n_sources_available:
            counter_loss = -q["on_prob"].log_prob(true_tile_cat["n_sources"])
            loss = counter_loss

        # all the squeezing/rearranging below is because a TileCatalog can store multiple
        # light sources per tile, which is annoying here, but helpful for storing samples
        # and real catalogs. Still, there may be a better way.

        # location loss
        if self.loc_available:
            true_locs = true_tile_cat["locs"].squeeze(3)
            locs_loss = -q["loc"].log_prob(true_locs)
            locs_loss *= true_tile_cat["n_sources"]
            loss += locs_loss

        # star/galaxy classification loss
        if self.source_type_available:
            true_gal_bools = rearrange(true_tile_cat.galaxy_bools, "b ht wt 1 1 -> b ht wt")
            binary_loss = -q["galaxy_prob"].log_prob(true_gal_bools)
            binary_loss *= true_tile_cat["n_sources"]
            loss += binary_loss

        # flux losses
        star_galaxy_flux_factors_available = self.star_flux_available and self.galaxy_flux_available
        if star_galaxy_flux_factors_available:
            true_star_bools = rearrange(true_tile_cat.star_bools, "b ht wt 1 1 -> b ht wt")
            star_fluxes = rearrange(true_tile_cat["star_fluxes"], "b ht wt 1 bnd -> b ht wt bnd")
            galaxy_fluxes = rearrange(
                true_tile_cat["galaxy_fluxes"], "b ht wt 1 bnd -> b ht wt bnd"
            )

            # only compute loss over bands we're using
            sg_flux_name_list = zip(self.star_flux_name_lst, self.galaxy_flux_name_lst)
            for i, (star_flux_factor, galaxy_flux_factor) in enumerate(sg_flux_name_list):
                # star flux loss
                star_flux_loss = (
                    -q[star_flux_factor].log_prob(star_fluxes[..., i] + 1e-9) * true_star_bools
                )
                loss += star_flux_loss

                # galaxy flux loss
                gal_flux_loss = (
                    -q[galaxy_flux_factor].log_prob(galaxy_fluxes[..., i] + 1e-9) * true_gal_bools
                )
                loss += gal_flux_loss

        # galaxy properties loss
        if self.galaxy_params_available:
            galsim_true_vals = rearrange(true_tile_cat["galaxy_params"], "b ht wt 1 d -> b ht wt d")
            for i, galaxy_params_factor in enumerate(self.galaxy_params_name_lst):
                loss_term = (
                    -q[galaxy_params_factor].log_prob(galsim_true_vals[..., i] + 1e-9)
                    * true_gal_bools
                )
                loss += loss_term

        return loss
