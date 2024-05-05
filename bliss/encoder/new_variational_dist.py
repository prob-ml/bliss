import torch
from einops import rearrange

from bliss.catalog import SourceType, TileCatalog
from bliss.encoder.unconstrained_dists import (
    UnconstrainedBernoulli,
    UnconstrainedLogitNormal,
    UnconstrainedLogNormal,
    UnconstrainedTDBN,
)


class NewVariationalDistSpec(torch.nn.Module):
    def __init__(self, survey_bands, tile_slen):
        super().__init__()

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen

        self.factor_specs = {
            "on_prob": UnconstrainedBernoulli(),
            "loc": UnconstrainedTDBN(),
            "galaxy_prob": UnconstrainedBernoulli(),
            # galsim parameters
            "galsim_disk_frac": UnconstrainedLogitNormal(),
            "galsim_beta_radians": UnconstrainedLogitNormal(high=torch.pi),
            "galsim_disk_q": UnconstrainedLogitNormal(),
            "galsim_a_d": UnconstrainedLogNormal(),
            "galsim_bulge_q": UnconstrainedLogitNormal(),
            "galsim_a_b": UnconstrainedLogNormal(),
            "galsim_disk_hlr": UnconstrainedLogNormal(),
            "galsim_bulge_hlr": UnconstrainedLogNormal(),
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
        return NewVariationalDist(factors, self.survey_bands, self.tile_slen)


class NewVariationalDist(torch.nn.Module):
    GALSIM_NAMES = [
        "disk_frac", "beta_radians", "disk_q", "a_d", "bulge_q", "a_b", "disk_hlr", "bulge_hlr"
    ]

    def __init__(self, factors, survey_bands, tile_slen):
        super().__init__()

        self.factors = factors
        self.survey_bands = survey_bands
        self.tile_slen = tile_slen

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
        gs_dists = []
        for name in self.GALSIM_NAMES:
            if f"galsim_{name}" in q:
                gs_dists.append(q[f"galsim_{name}"])
            else:
                gs_dists.append(None)

        gs_param_lst = []
        for d in gs_dists:
            if d:
                gs_param_lst.append(d.icdf(torch.tensor(0.5)) if use_mode else d.sample())
            else:
                gs_param_lst.append(torch.zeros_like(gs_param_lst[0]))
        est_cat["galaxy_params"] = torch.stack(gs_param_lst, dim=3)

        # populate catalog with per-band galaxy fluxes
        gf_dists = [q[f"galaxy_flux_{band}"] for band in self.survey_bands]
        gf_lst = [d.icdf(torch.tensor(0.5)) if use_mode else d.sample() for d in gf_dists]
        est_cat["galaxy_fluxes"] = torch.stack(gf_lst, dim=3)

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

        # counter loss
        counter_loss = -q["on_prob"].log_prob(true_tile_cat.n_sources)
        loss = counter_loss

        # all the squeezing/rearranging below is because a TileCatalog can store multiple
        # light sources per tile, which is annoying here, but helpful for storing samples
        # and real catalogs. Still, there may be a better way.

        # location loss
        true_locs = true_tile_cat.locs.squeeze(3)
        locs_loss = -q["loc"].log_prob(true_locs)
        locs_loss *= true_tile_cat.n_sources
        loss += locs_loss

        # star/galaxy classification loss
        true_gal_bools = rearrange(true_tile_cat.galaxy_bools, "b ht wt 1 1 -> b ht wt")
        binary_loss = -q["galaxy_prob"].log_prob(true_gal_bools)
        binary_loss *= true_tile_cat.n_sources
        loss += binary_loss

        # flux losses
        true_star_bools = rearrange(true_tile_cat.star_bools, "b ht wt 1 1 -> b ht wt")
        star_fluxes = rearrange(true_tile_cat["star_fluxes"], "b ht wt 1 bnd -> b ht wt bnd")
        galaxy_fluxes = rearrange(true_tile_cat["galaxy_fluxes"], "b ht wt 1 bnd -> b ht wt bnd")

        # only compute loss over bands we're using
        for i, band in enumerate(self.survey_bands):
            # star flux loss
            star_name = f"star_flux_{band}"
            star_flux_loss = -q[star_name].log_prob(star_fluxes[..., i] + 1e-9) * true_star_bools
            loss += star_flux_loss

            # galaxy flux loss
            gal_name = f"galaxy_flux_{band}"
            gal_flux_loss = -q[gal_name].log_prob(galaxy_fluxes[..., i] + 1e-9) * true_gal_bools
            loss += gal_flux_loss

        # galaxy properties loss
        galsim_true_vals = rearrange(true_tile_cat["galaxy_params"], "b ht wt 1 d -> b ht wt d")
        for i, param_name in enumerate(self.GALSIM_NAMES):
            galsim_pn = f"galsim_{param_name}"
            if galsim_pn in self.factors:  # only apply over factors we're using
                loss_term = -q[galsim_pn].log_prob(galsim_true_vals[..., i] + 1e-9) * true_gal_bools
                loss += loss_term

        return loss
