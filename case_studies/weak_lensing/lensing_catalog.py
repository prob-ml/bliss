import torch
from einops import rearrange, reduce, repeat

from bliss.catalog import FullCatalog, TileCatalog


class LensingTileCatalog(TileCatalog):
    allowed_params = {
        "n_source_log_probs",
        "fluxes",
        "star_fluxes",
        "star_log_fluxes",
        "mags",
        "ellips",
        "snr",
        "blendedness",
        "source_type",
        "galaxy_params",
        "galaxy_fluxes",
        "galaxy_probs",
        "galaxy_blends",
        "objid",
        "hlr",
        "ra",
        "dec",
        "matched",
        "mismatched",
        "detection_thresholds",
        "log_flux_sd",
        "loc_sd",
    }

    def to_full_catalog(self):
        """Converts image parameters in tiles to parameters of full image.

        By parameters, we mean samples from the variational distribution, not the variational
        parameters.

        Returns:
            LensingFullCatalog with the same specification.
        """
        plocs = self.get_full_locs_from_tiles()
        param_names_to_mask = {"plocs"}.union(set(self.keys()))
        tile_params_to_gather = {"plocs": plocs}
        tile_params_to_gather.update(self)

        params = {}
        indices_to_retrieve, is_on_array = self.get_indices_of_on_sources()
        for param_name, tile_param in tile_params_to_gather.items():
            k = tile_param.shape[-1]
            param = rearrange(tile_param, "b nth ntw s k -> b (nth ntw s) k", k=k)
            indices_for_param = repeat(indices_to_retrieve, "b nth_ntw_s -> b nth_ntw_s k", k=k)
            param = torch.gather(param, dim=1, index=indices_for_param)
            if param_name in param_names_to_mask:
                param = param * is_on_array.unsqueeze(-1)
            params[param_name] = param

        params["n_sources"] = reduce(self.n_sources, "b nth ntw -> b", "sum")
        return LensingFullCatalog(self.height, self.width, params)

    def __repr__(self):
        return f"LensingTileCatalog({self.batch_size} x {self.n_tiles_h} x {self.n_tiles_w})"


class LensingFullCatalog(FullCatalog):
    allowed_params = LensingTileCatalog.allowed_params

    def __repr__(self):
        return "LensingFullCatalog"
