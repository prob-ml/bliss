import copy

import torch
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
    def sample_first_detection(self, x_features, use_mode=True, true_tile_cat=None):
        batch_size, _n_features, ht, wt = x_features.shape[0:4]

        est_cat = None
        patterns_to_use = (0, 8, 12, 14) if self.use_checkerboard else (0,)

        for mask_pattern in self.mask_patterns[patterns_to_use, ...]:
            mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
            context1 = self.make_context(est_cat, mask)
            x_cat1 = self.catalog_net(x_features, context1)
            new_est_cat = self.var_dist.sample(
                x_cat1, use_mode=use_mode, true_tile_cat=true_tile_cat
            )
            new_est_cat["n_sources"] *= 1 - mask
            if est_cat is None:
                est_cat = new_est_cat
            else:
                est_cat["n_sources"] *= mask
                est_cat = est_cat.union(new_est_cat, disjoint=True)

        return est_cat

    def sample_second_detection(self, x_features, est_cat1, use_mode=True, true_tile_cat=None):
        no_mask = torch.ones_like(est_cat1["n_sources"])
        context2 = self.make_context(est_cat1, no_mask, detection2=True)
        x_cat2 = self.catalog_net(x_features, context2)
        est_cat2 = self.var_dist.sample(x_cat2, use_mode=use_mode, true_tile_cat=true_tile_cat)
        # our loss function implies that the second detection is ignored for a tile
        # if the first detection is empty for that tile
        est_cat2["n_sources"] *= est_cat1["n_sources"]
        return est_cat2

    def sample(self, batch, use_mode=True):
        assert isinstance(self.var_dist, CalibrationVariationalDist)

        x_features = self.get_features(batch)
        true_tile_dict = batch.get("tile_catalog", None)
        if true_tile_dict is not None:
            true_tile_cat = TileCatalog(true_tile_dict)
            true_tile_cat1 = true_tile_cat.get_brightest_sources_per_tile(
                top_k=1,
                exclude_num=0,
                band=2,
            )
        else:
            true_tile_cat1 = None
        est_cat = self.sample_first_detection(
            x_features,
            use_mode=use_mode,
            true_tile_cat=true_tile_cat1,
        )

        if self.use_double_detect:
            if true_tile_dict is not None:
                true_tile_cat2 = true_tile_cat.get_brightest_sources_per_tile(
                    top_k=1,
                    exclude_num=1,
                    band=2,
                )
            else:
                true_tile_cat2 = None
            est_cat2 = self.sample_second_detection(
                x_features,
                est_cat,
                use_mode=use_mode,
                true_tile_cat=true_tile_cat2,
            )
            est_cat = est_cat.union(est_cat2, disjoint=False)

        return self._add_source_mask(est_cat)
