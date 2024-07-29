import copy

import torch
from einops import rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder


class MyEncoder(Encoder):
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

    def sample_vsbc_first_detection(self, x_features, true_tile_cat, use_mode=True):
        batch_size, _n_features, ht, wt = x_features.shape[0:4]

        est_vsbc_cat = None
        patterns_to_use = (0, 8, 12, 14) if self.use_checkerboard else (0,)

        for mask_pattern in self.mask_patterns[patterns_to_use, ...]:
            mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
            context1 = self.make_context(est_vsbc_cat, mask)
            x_cat1 = self.catalog_net(x_features, context1)
            new_est_vsbc_cat = self.var_dist.sample_vsbc(x_cat1, true_tile_cat, use_mode=use_mode)
            new_est_vsbc_cat["n_sources"] *= 1 - mask
            if est_vsbc_cat is None:
                est_vsbc_cat = new_est_vsbc_cat
            else:
                est_vsbc_cat["n_sources"] *= mask
                est_vsbc_cat = est_vsbc_cat.union(new_est_vsbc_cat, disjoint=True)

        return est_vsbc_cat

    def sample_vsbc_second_detection(self, x_features, est_cat1, true_tile_cat, use_mode=True):
        no_mask = torch.ones_like(est_cat1["n_sources"])
        context2 = self.make_context(est_cat1, no_mask, detection2=True)
        x_cat2 = self.catalog_net(x_features, context2)
        est_vsbc_cat2 = self.var_dist.sample_vsbc(x_cat2, true_tile_cat, use_mode=use_mode)
        est_vsbc_cat2["n_sources"] *= est_cat1["n_sources"]
        return est_vsbc_cat2

    def sample_vsbc(self, batch, use_mode=True):
        true_tile_cat = TileCatalog(batch["tile_catalog"])
        x_features = self.get_features(batch)
        est_vsbc_cat = self.sample_vsbc_first_detection(
            x_features, true_tile_cat, use_mode=use_mode
        )
        if self.use_double_detect:
            est_vsbc_cat2 = self.sample_vsbc_second_detection(
                x_features, est_vsbc_cat, true_tile_cat, use_mode=use_mode
            )
            est_vsbc_cat = est_vsbc_cat.union(est_vsbc_cat2, disjoint=False)
        return est_vsbc_cat
