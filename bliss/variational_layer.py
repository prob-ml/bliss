import torch
from einops import rearrange

from bliss.catalog import SourceType, TileCatalog


class VariationalLayer(torch.nn.Module):
    GALSIM_NAMES = ["disk_frac", "beta_radians", "disk_q", "a_d", "bulge_q", "a_b"]

    def __init__(self, pred, survey_bands, tile_slen) -> None:
        super().__init__()
        self.pred = pred
        self.survey_bands = survey_bands
        self.tile_slen = tile_slen

    def sample(self, use_mode=False) -> TileCatalog:
        """Sample the variational distribution.

        Args:
            use_mode: whether to use the mode of the distribution instead of random sampling

        Returns:
            TileCatalog: Sampled catalog
        """
        pred = self.pred

        locs = pred["loc"].mode if use_mode else pred["loc"].sample().squeeze(0)
        est_cat = {"locs": locs}

        # populate catalog with per-band (log) star fluxes
        sf_preds = [pred[f"star_flux_{band}"] for band in self.survey_bands]
        sf_lst = [p.mode if use_mode else p.sample() for p in sf_preds]
        est_cat["star_fluxes"] = torch.stack(sf_lst, dim=3)

        # populate catalog with source type
        galaxy_bools = pred["galaxy_prob"].mode if use_mode else pred["galaxy_prob"].sample()
        galaxy_bools = galaxy_bools.unsqueeze(3)
        star_bools = 1 - galaxy_bools
        est_cat["source_type"] = SourceType.STAR * star_bools + SourceType.GALAXY * galaxy_bools

        # populate catalog with galaxy parameters
        gs_dists = [pred[f"galsim_{name}"] for name in self.GALSIM_NAMES]
        gs_param_lst = [d.icdf(torch.tensor(0.5)) if use_mode else d.sample() for d in gs_dists]
        est_cat["galaxy_params"] = torch.stack(gs_param_lst, dim=3)

        # populate catalog with per-band galaxy fluxes
        gf_dists = [pred[f"galaxy_flux_{band}"] for band in self.survey_bands]
        gf_lst = [d.icdf(torch.tensor(0.5)) if use_mode else d.sample() for d in gf_dists]
        est_cat["galaxy_fluxes"] = torch.stack(gf_lst, dim=3)

        # we have to unsqueeze these tensors because a TileCatalog can store multiple
        # light sources per tile, but we sample only one source per tile
        for k, v in est_cat.items():
            est_cat[k] = v.unsqueeze(3)

        # n_sources is not unsqueezed because it is a single integer per tile regardless of
        # how many light sources are stored per tile
        est_cat["n_sources"] = pred["on_prob"].mode if use_mode else pred["on_prob"].sample()

        return TileCatalog(self.tile_slen, est_cat)

    def compute_nll(self, true_tile_cat: TileCatalog, tile_mask: torch.Tensor) -> dict:
        pred = self.pred

        loss_with_components = {}

        # counter loss
        counter_loss = -pred["on_prob"].log_prob(true_tile_cat.n_sources)
        counter_loss *= tile_mask
        loss = counter_loss
        loss_with_components["counter_loss"] = counter_loss.sum()

        # all the squeezing/rearranging below is because a TileCatalog can store multiple
        # light sources per tile, which is annoying here, but helpful for storing samples
        # and real catalogs. Still, there may be a better way.

        # location loss
        true_locs = true_tile_cat.locs.squeeze(3)
        locs_loss = -pred["loc"].log_prob(true_locs)
        locs_loss *= true_tile_cat.n_sources
        locs_loss *= tile_mask
        loss += locs_loss
        loss_with_components["locs_loss"] = locs_loss.sum()

        # star/galaxy classification loss
        true_gal_bools = rearrange(true_tile_cat.galaxy_bools, "b ht wt 1 1 -> b ht wt")
        binary_loss = -pred["galaxy_prob"].log_prob(true_gal_bools)
        binary_loss *= true_tile_cat.n_sources
        binary_loss *= tile_mask
        loss += binary_loss
        loss_with_components["binary_loss"] = binary_loss.sum()

        # flux losses
        true_star_bools = rearrange(true_tile_cat.star_bools, "b ht wt 1 1 -> b ht wt")
        star_fluxes = rearrange(true_tile_cat["star_fluxes"], "b ht wt 1 bnd -> b ht wt bnd")
        galaxy_fluxes = rearrange(true_tile_cat["galaxy_fluxes"], "b ht wt 1 bnd -> b ht wt bnd")

        # only compute loss over bands we're using
        for i, band in enumerate(self.survey_bands):
            # star flux loss
            star_name = f"star_flux_{band}"
            star_flux_loss = -pred[star_name].log_prob(star_fluxes[..., i] + 1e-9) * true_star_bools
            star_flux_loss *= tile_mask
            loss_with_components[star_name] = star_flux_loss.sum()
            loss += star_flux_loss

            # galaxy flux loss
            gal_name = f"galaxy_flux_{band}"
            gal_flux_loss = -pred[gal_name].log_prob(galaxy_fluxes[..., i] + 1e-9) * true_gal_bools
            gal_flux_loss *= tile_mask
            loss_with_components[gal_name] = gal_flux_loss.sum()
            loss += gal_flux_loss

        # galaxy properties loss
        galsim_true_vals = rearrange(true_tile_cat["galaxy_params"], "b ht wt 1 d -> b ht wt d")
        for i, param_name in enumerate(self.GALSIM_NAMES):
            galsim_pn = f"galsim_{param_name}"
            loss_term = -pred[galsim_pn].log_prob(galsim_true_vals[..., i] + 1e-9) * true_gal_bools
            loss_term *= tile_mask
            loss_with_components[galsim_pn] = loss_term.sum()
            loss += loss_term

        # we really shouldn't normalize this loss by the number of sources if we're subsequently
        # summing it with loss from other layers
        loss_with_components["loss"] = loss.sum()

        return loss_with_components
