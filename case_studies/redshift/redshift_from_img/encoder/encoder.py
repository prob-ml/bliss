from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder


class RedshiftsEncoder(Encoder):
    def update_metrics(self, batch, batch_idx):
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])
        target_cat = target_cat.filter_tile_catalog_by_flux(
            min_flux=self.min_flux_for_loss,
            band=self.reference_band,
        )
        target_cat = target_cat.symmetric_crop(self.tiles_to_crop)

        mode_cat = self.sample(batch, use_mode=True)
        matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.mode_metrics.update(target_cat, mode_cat, matching)

        sample_cat = self.sample(batch, use_mode=False)
        matching = self.matcher.match_catalogs(target_cat, sample_cat)
        self.sample_metrics.update(target_cat, sample_cat, matching)

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/2529
        self.mode_metrics = self.mode_metrics.to(self.device)
        self.sample_metrics = self.sample_metrics.to(self.device)
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.report_metrics(self.sample_metrics, "val/sample", show_epoch=True)
