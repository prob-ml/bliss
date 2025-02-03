import torch

from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder


class RedshiftsEncoder(Encoder):
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
        batch_size, _n_bands, h, w = batch["images"].shape[0:4]
        ht, wt = h // self.tile_slen, w // self.tile_slen

        input_lst = [
            inorm.get_input_tensor(batch).to(batch["images"].device)
            for inorm in self.image_normalizers
        ]
        x = torch.cat(input_lst, dim=2)
        x_features = self.features_net(x)
        mask = torch.zeros([batch_size, ht, wt])
        context = self.make_context(None, mask).to("cuda")
        x_cat_marginal = self.catalog_net(x_features, context)
        return x_features, x_cat_marginal

    def sample(self, batch, use_mode=True):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.sample(x_cat_marginal, use_mode=use_mode)

    def discrete_sample(self, batch, use_mode=True, risk_type=None):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.discrete_sample(x_cat_marginal, use_mode=use_mode, risk_type=risk_type)
