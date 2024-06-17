from bliss.catalog import BaseTileCatalog
from bliss.encoder.encoder import Encoder


class RedshiftsEncoder(Encoder):
    def update_metrics(self, batch, batch_idx):
        target_cat = BaseTileCatalog(self.tile_slen, batch["tile_catalog"])
        target_cat = target_cat.symmetric_crop(self.tiles_to_crop)

        mode_cat = self.sample(batch, use_mode=True)
        self.mode_metrics.update(target_cat, mode_cat)

        sample_cat = self.sample(batch, use_mode=False)
        self.sample_metrics.update(target_cat, sample_cat)

    def on_validation_epoch_end(self):
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.report_metrics(self.sample_metrics, "val/sample", show_epoch=True)
