from typing import List

import torch
from einops import rearrange

from bliss.catalog import TileCatalog
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
            "n_sources": UnconstrainedBernoulli(),
            "locs": UnconstrainedTDBN(),
            "source_type": UnconstrainedBernoulli(),
            "star_fluxes": UnconstrainedLogNormal(dim=len(survey_bands)),
            "galaxy_fluxes": UnconstrainedLogNormal(dim=len(survey_bands)),
            # galsim parameters
            "galsim_disk_frac": UnconstrainedLogitNormal(),
            "galsim_beta_radians": UnconstrainedLogitNormal(high=torch.pi),
            "galsim_disk_q": UnconstrainedLogitNormal(),
            "galsim_a_d": UnconstrainedLogNormal(),
            "galsim_bulge_q": UnconstrainedLogitNormal(),
            "galsim_a_b": UnconstrainedLogNormal(),
        }

    @property
    def n_params_per_source(self):
        return sum(fs.n_params for fs in self.factor_specs.values())

    def _parse_factors(self, x_cat):
        split_sizes = [v.n_params for v in self.factor_specs.values()]
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
        return VariationalDist(factors, self.tile_slen)


class VariationalDist(torch.nn.Module):
    GALSIM_NAMES = ["disk_frac", "beta_radians", "disk_q", "a_d", "bulge_q", "a_b"]

    def __init__(self, factors, tile_slen):
        super().__init__()

        self.factors = factors
        self.tile_slen = tile_slen
        self.factors_name_set = set(self.factors.keys())

        self.loc_name_lst = ["locs"]
        self.star_flux_name_lst = ["star_fluxes"]
        self.source_type_name_lst = ["source_type"]
        self.galsim_params_name_lst = [f"galsim_{name}" for name in self.GALSIM_NAMES]
        self.galaxy_flux_name_lst = ["galaxy_fluxes"]
        self.n_sources_name_lst = ["n_sources"]

        self.loc_available = self._factors_are_available(self.loc_name_lst)
        self.star_flux_available = self._factors_are_available(self.star_flux_name_lst)
        self.source_type_available = self._factors_are_available(self.source_type_name_lst)
        self.galaxy_params_available = self._factors_are_available(self.galsim_params_name_lst)
        self.galaxy_flux_available = self._factors_are_available(self.galaxy_flux_name_lst)
        self.n_sources_available = self._factors_are_available(self.n_sources_name_lst)

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

        # populate catalog with locations
        if self.loc_available:
            est_cat["locs"] = q["locs"].mode if use_mode else q["locs"].sample()

        # populate catalog with source type
        if self.source_type_available:
            est_cat["source_type"] = (
                q["source_type"].mode if use_mode else q["source_type"].sample()
            ).unsqueeze(-1)

        # populate catalog with per-band (log) star fluxes
        if self.star_flux_available:
            est_cat["star_fluxes"] = (
                q["star_fluxes"].mode if use_mode else q["star_fluxes"].sample()
            )

        # populate catalog with per-band galaxy fluxes
        if self.galaxy_flux_available:
            est_cat["galaxy_fluxes"] = (
                q["galaxy_fluxes"].mode if use_mode else q["galaxy_fluxes"].sample()
            )

        # populate catalog with galaxy parameters
        if self.galaxy_params_available:
            gs_dists = [q[factor] for factor in self.galsim_params_name_lst]
            gs_param_lst = [d.mode if use_mode else d.sample() for d in gs_dists]
            est_cat["galaxy_params"] = torch.concat(gs_param_lst, dim=3)

        # we have to unsqueeze these tensors because a TileCatalog can store multiple
        # light sources per tile, but we sample only one source per tile
        for k, v in est_cat.items():
            est_cat[k] = v.unsqueeze(3)

        # n_sources is not unsqueezed because it is a single integer per tile regardless of
        # how many light sources are stored per tile
        if self.n_sources_available:
            est_cat["n_sources"] = q["n_sources"].mode if use_mode else q["n_sources"].sample()

        return TileCatalog(self.tile_slen, est_cat)

    def compute_nll(self, true_tile_cat: TileCatalog):
        q = self.factors

        # counter loss
        if self.n_sources_available:
            counter_loss = -q["n_sources"].log_prob(true_tile_cat["n_sources"])
            loss = counter_loss

        # all the squeezing/rearranging below is because a TileCatalog can store multiple
        # light sources per tile, which is annoying here, but helpful for storing samples
        # and real catalogs. Still, there may be a better way.

        # location loss
        if self.loc_available:
            true_locs = true_tile_cat["locs"].squeeze(3)
            locs_loss = -q["locs"].log_prob(true_locs)
            locs_loss *= true_tile_cat["n_sources"]
            loss += locs_loss

        # star/galaxy classification loss
        if self.source_type_available:
            true_gal_bools = rearrange(true_tile_cat.galaxy_bools, "b ht wt 1 1 -> b ht wt")
            binary_loss = -q["source_type"].log_prob(true_gal_bools)
            binary_loss *= true_tile_cat["n_sources"]
            loss += binary_loss

        # star flux loss
        if self.star_flux_available:
            true_star_bools = rearrange(true_tile_cat.star_bools, "b ht wt 1 1 -> b ht wt")
            star_fluxes = rearrange(true_tile_cat["star_fluxes"], "b ht wt 1 bnd -> b ht wt bnd")
            star_flux_loss = -q["star_fluxes"].log_prob(star_fluxes + 1e-9) * true_star_bools
            loss += star_flux_loss

        # galaxy flux loss
        if self.galaxy_flux_available:
            gal_fluxes = rearrange(true_tile_cat["galaxy_fluxes"], "b ht wt 1 bnd -> b ht wt bnd")
            gal_flux_loss = -q["galaxy_fluxes"].log_prob(gal_fluxes + 1e-9) * true_gal_bools
            loss += gal_flux_loss

        # galaxy properties loss
        if self.galaxy_params_available:
            for i, gs_param in enumerate(self.galsim_params_name_lst):
                gs_true = true_tile_cat["galaxy_params"][..., i]
                loss_term = -q[gs_param].log_prob(gs_true + 1e-9) * true_gal_bools
                loss += loss_term

        return loss
