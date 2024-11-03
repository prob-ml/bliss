import copy

from einops import rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder
from case_studies.dc2_cataloging.utils.variational_dist import CalibrationVariationalDist


class MyBasicEncoder(Encoder):
    @classmethod
    def _add_source_mask(cls, ori_tile_cat: TileCatalog):
        d = copy.copy(ori_tile_cat.data)
        on_mask = rearrange(ori_tile_cat.is_on_mask, "b nth ntw s -> b nth ntw s 1")
        on_mask_count = on_mask.sum(dim=(-2, -1))
        d["one_source_mask"] = rearrange(on_mask_count == 1, "b nth ntw -> b nth ntw 1 1") & on_mask
        d["two_sources_mask"] = (
            rearrange(on_mask_count == 2, "b nth ntw -> b nth ntw 1 1") & on_mask
        )
        d["more_than_two_sources_mask"] = (
            rearrange(on_mask_count > 2, "b nth ntw -> b nth ntw 1 1") & on_mask
        )
        return TileCatalog(d)

    def sample(self, batch, use_mode=True):
        tile_cat = super().sample(batch, use_mode)
        return self._add_source_mask(tile_cat)


class CalibrationEncoder(MyBasicEncoder):
    def sample(self, batch, use_mode=True):
        assert isinstance(self.var_dist, CalibrationVariationalDist)

        true_tile_dict = batch.get("tile_catalog", None)
        if true_tile_dict is not None:
            sample_context = []
            true_tile_cat = TileCatalog(true_tile_dict)
            true_tile_cat1 = true_tile_cat.get_brightest_sources_per_tile(
                top_k=1,
                exclude_num=0,
                band=2,
            )
            sample_context.append(true_tile_cat1)
            if self.use_double_detect:
                true_tile_cat2 = true_tile_cat.get_brightest_sources_per_tile(
                    top_k=1,
                    exclude_num=1,
                    band=2,
                )
                sample_context.append(true_tile_cat2)
            self.var_dist.init_sample_context(sample_context)
        else:
            self.var_dist.clear_sample_context()

        est_cat = super().sample(batch, use_mode)

        self.var_dist.clear_sample_context()

        return est_cat
