import copy

import torch
from einops import rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder


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
    def sample_first_detection(self, x_features, use_mode=True, sample_func=None):
        if sample_func is None:
            sample_func = lambda var_dist, x_cat, use_mode: var_dist.sample(
                x_cat, use_mode=use_mode
            )

        batch_size, _n_features, ht, wt = x_features.shape[0:4]

        est_cat = None
        patterns_to_use = (0, 8, 12, 14) if self.use_checkerboard else (0,)

        for mask_pattern in self.mask_patterns[patterns_to_use, ...]:
            mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
            context1 = self.make_context(est_cat, mask)
            x_cat1 = self.catalog_net(x_features, context1)
            new_est_cat = sample_func(self.var_dist, x_cat1, use_mode=use_mode)
            new_est_cat["n_sources"] *= 1 - mask
            if est_cat is None:
                est_cat = new_est_cat
            else:
                est_cat["n_sources"] *= mask
                est_cat = est_cat.union(new_est_cat, disjoint=True)

        return est_cat

    def sample_second_detection(self, x_features, est_cat1, use_mode=True, sample_func=None):
        if sample_func is None:
            sample_func = lambda var_dist, x_cat, use_mode: var_dist.sample(
                x_cat, use_mode=use_mode
            )
        no_mask = torch.ones_like(est_cat1["n_sources"])
        context2 = self.make_context(est_cat1, no_mask, detection2=True)
        x_cat2 = self.catalog_net(x_features, context2)
        est_cat2 = sample_func(self.var_dist, x_cat2, use_mode=use_mode)
        # our loss function implies that the second detection is ignored for a tile
        # if the first detection is empty for that tile
        est_cat2["n_sources"] *= est_cat1["n_sources"]
        return est_cat2

    @torch.no_grad()
    def sample_vsbc(self, batch, use_mode=True):
        true_tile_cat = TileCatalog(batch["tile_catalog"])
        x_features = self.get_features(batch)
        true_tile_cat1 = true_tile_cat.get_brightest_sources_per_tile(
            top_k=1, exclude_num=0, band=2
        )
        first_sample_func = lambda var_dist, x_cat, use_mode: var_dist.sample_vsbc(
            x_cat, true_tile_cat1, use_mode=use_mode
        )
        est_vsbc_cat = self.sample_first_detection(
            x_features, use_mode=use_mode, sample_func=first_sample_func
        )
        if self.use_double_detect:
            true_tile_cat2 = true_tile_cat.get_brightest_sources_per_tile(
                top_k=1, exclude_num=1, band=2
            )
            second_sample_func = lambda var_dist, x_cat, use_mode: var_dist.sample_vsbc(
                x_cat, true_tile_cat2, use_mode=use_mode
            )
            est_vsbc_cat2 = self.sample_second_detection(
                x_features, est_vsbc_cat, use_mode=use_mode, sample_func=second_sample_func
            )
            est_vsbc_cat = est_vsbc_cat.union(est_vsbc_cat2, disjoint=False)
        return est_vsbc_cat

    @torch.no_grad()
    def sample_credible_interval(self, batch, lower_q, upper_q, use_mode=True):
        x_features = self.get_features(batch)
        sample_func = lambda var_dist, x_cat, use_mode: var_dist.sample_credible_interval(
            x_cat, lower_q=lower_q, upper_q=upper_q, use_mode=use_mode
        )
        est_ci_cat = self.sample_first_detection(
            x_features, use_mode=use_mode, sample_func=sample_func
        )
        if self.use_double_detect:
            est_ci_cat2 = self.sample_second_detection(
                x_features, est_ci_cat, use_mode=use_mode, sample_func=sample_func
            )
            est_ci_cat = est_ci_cat.union(est_ci_cat2, disjoint=False)
        return est_ci_cat
