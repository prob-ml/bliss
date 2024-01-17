from bliss.encoder.encoder import Encoder
from bliss.encoder.unconstrained_dists import UnconstrainedNormal


class RedshiftEncoder(Encoder):
    """New encoder class.
    New distribution for redshift per-source in
    dist_param_groups.
    """

    @property
    def dist_param_groups(self):
        d = super().dist_param_groups
        d["redshift"] = UnconstrainedNormal()
        return d
