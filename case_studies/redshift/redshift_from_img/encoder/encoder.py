import torch

from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder


class RedshiftsEncoder(Encoder):
    def __init__(
        self,
        discrete_metrics,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.discrete_metrics = discrete_metrics

    def update_metrics(self, batch, batch_idx):
        target_cat = TileCatalog(batch["tile_catalog"]).get_brightest_sources_per_tile()

        for risk_type in self.discrete_metrics.keys():
            mode_cat = self.discrete_sample(batch, use_mode=True, risk_type=risk_type)
            matching = self.matcher.match_catalogs(target_cat, mode_cat)
            self.discrete_metrics[risk_type].update(target_cat, mode_cat, matching)

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

    def get_features_and_parameters(self, batch):
        batch = (
            batch
            if isinstance(batch, dict)
            else {"images": batch, "background": torch.zeros_like(batch)}
        )
        x_features = self.get_features(batch)
        batch_size, _n_features, ht, wt = x_features.shape[0:4]
        pattern_to_use = (0,)  # no checkerboard
        mask_pattern = self.mask_patterns[pattern_to_use, ...]
        est_cat = None
        history_mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
        x_color_context = self.make_color_context(est_cat, history_mask)
        x_features_color = torch.cat((x_features, x_color_context), dim=1)
        x_cat_marginal = self.detect_first(x_features_color)

        return x_features_color, x_cat_marginal

    def sample(self, batch, use_mode=True):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.sample(x_cat_marginal, use_mode=use_mode)

    def discrete_sample(self, batch, use_mode=True, risk_type=None):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.discrete_sample(x_cat_marginal, use_mode=use_mode, risk_type=risk_type)
