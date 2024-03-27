import torch
from einops import rearrange, reduce, repeat

from bliss.catalog import FullCatalog, TileCatalog


class RedshiftTileCatalog(TileCatalog):
    allowed_params = TileCatalog.allowed_params  # ideally don't alter TileCatalog

    def to_full_catalog(self):
        """Copy/paste from super, but return RedshiftFullCatalog.
        TODO: fewer lines of repeated code.

        Returns:
            RedshiftFullCatalog of same specification.
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
        return RedshiftFullCatalog(self.height, self.width, params)

    def __repr__(self):
        return f"RedshiftTileCatalog({self.batch_size} x {self.n_tiles_h} x {self.n_tiles_w})"


class RedshiftFullCatalog(FullCatalog):
    allowed_params = RedshiftTileCatalog.allowed_params

    def __repr__(self):
        return "RedshiftFullCatalog"
