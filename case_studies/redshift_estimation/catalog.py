from bliss.catalog import TileCatalog


class RedshiftTileCatalog(TileCatalog):
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
        "redshifts",
    }

    def __init__(self, redshifts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redshifts = redshifts

    def __repr__(self):
        return f"RedshiftTileCatalog({self.batch_size} x {self.n_tiles_h} x {self.n_tiles_w})"
