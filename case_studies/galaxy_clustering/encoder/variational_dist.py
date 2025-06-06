from bliss.catalog import BaseTileCatalog
from bliss.encoder.variational_dist import VariationalDist


class GalaxyClusterVariationalDist(VariationalDist):
    def sample(self, x_cat, use_mode=True):
        fp_pairs = self._factor_param_pairs(x_cat)
        d = {qk.name: qk.sample(params, use_mode) for qk, params in fp_pairs}
        return BaseTileCatalog(d)
