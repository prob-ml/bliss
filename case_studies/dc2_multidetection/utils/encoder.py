import warnings
from typing import Optional

import torch
from einops import rearrange, repeat
from torchmetrics import MetricCollection

from bliss.catalog import TileCatalog
from bliss.encoder.metrics import CatalogMatcher
from case_studies.dc2_cataloging.utils.encoder import MyBasicEncoder
from case_studies.dc2_multidetection.utils.convnet import SimpleCatalogNet, SimpleFeaturesNet
from case_studies.dc2_multidetection.utils.variational_dist import MultiVariationalDist


class MultiDetectEncoder(MyBasicEncoder):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        image_normalizers: dict,
        var_dist: MultiVariationalDist,
        matcher: CatalogMatcher,
        sample_image_renders: MetricCollection,
        mode_metrics: MetricCollection,
        sample_metrics: Optional[MetricCollection] = None,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        use_double_detect: bool = False,
        use_checkerboard: bool = True,
        reference_band: int = 2,
        one_to_topk: int = 3,
    ):
        assert isinstance(var_dist, MultiVariationalDist), "var_dist should be MultiVariationalDist"
        super().__init__(
            survey_bands=survey_bands,
            tile_slen=tile_slen,
            image_normalizers=image_normalizers,
            var_dist=var_dist,
            matcher=matcher,
            sample_image_renders=sample_image_renders,
            mode_metrics=mode_metrics,
            sample_metrics=sample_metrics,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            use_double_detect=False,
            use_checkerboard=False,
            reference_band=reference_band,
        )
        assert one_to_topk > 1
        self.one_to_topk = one_to_topk

    def initialize_networks(self):
        assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        num_features = 256
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        assert ch_per_band == 1
        self.features_net = SimpleFeaturesNet(
            n_bands=len(self.survey_bands),
            ch_per_band=ch_per_band,
            num_features=num_features,
            double_downsample=(self.tile_slen == 4),
        )
        self.one_to_one_catalog_net = SimpleCatalogNet(
            num_features=num_features,
            out_channels=self.var_dist.n_params_per_source,
        )
        self.one_to_many_catalog_net = SimpleCatalogNet(
            num_features=num_features,
            out_channels=self.var_dist.n_params_per_source,
        )

    def make_context(self, history_cat, history_mask, detection2=False):
        raise NotImplementedError()

    def get_features(self, batch):
        assert batch["images"].size(2) % 16 == 0, "image dims must be multiples of 16"
        assert batch["images"].size(3) % 16 == 0, "image dims must be multiples of 16"
        assert len(self.image_normalizers) == 1

        network_input = self.image_normalizers[0].get_input_tensor(batch)
        network_input = rearrange(network_input, "b bands 1 h w -> b bands h w")
        return self.features_net(network_input)

    def sample_first_detection(self, x_features, use_mode=True):
        raise NotImplementedError()

    def sample_second_detection(self, x_features, est_cat1, use_mode=True):
        raise NotImplementedError()

    @classmethod
    def _accumulate_stacked_tile_cat(cls, stacked_tile_cat: TileCatalog):
        assert "n_sources_mask" in stacked_tile_cat
        _, accumulated_indices = torch.sort(
            stacked_tile_cat["n_sources_mask"].to(dtype=torch.int8), dim=-2, descending=True
        )  # (b, nth, ntw, m, 1)
        d = {}
        for k, v in stacked_tile_cat.items():
            if k == "n_sources":
                d[k] = v
            else:
                expanded_accumulated_indices = accumulated_indices.expand_as(v)
                d[k] = torch.gather(v, dim=-2, index=expanded_accumulated_indices)
        return TileCatalog(d)

    @classmethod
    def _choose_topk(cls, est_cat: TileCatalog, topk: int):
        assert topk >= 1
        n_sources_probs = est_cat["n_sources_probs"][..., 1:2]  # yes_probs (b, nth, ntw, m, 1)
        _, topk_indices = torch.topk(
            n_sources_probs, k=topk, dim=-2, largest=True
        )  # (b, nth, ntw, topk, 1)
        d = {}
        for k, v in est_cat.items():
            if k != "n_sources":
                repeated_topk_indices = repeat(
                    topk_indices, "b nth ntw topk 1 -> b nth ntw topk (d 1)", d=v.shape[-1]
                )
                d[k] = torch.gather(v, dim=-2, index=repeated_topk_indices)
            else:
                repeated_topk_indices = repeat(
                    topk_indices,
                    "b nth ntw topk 1 -> b nth ntw topk (d 1)",
                    d=est_cat["n_sources_mask"].shape[-1],
                )
                d["n_sources"] = torch.gather(
                    est_cat["n_sources_mask"], dim=-2, index=repeated_topk_indices
                ).sum(dim=(-2, -1))
        return TileCatalog(d)

    @torch.no_grad()
    def sample(self, batch, use_mode=True):
        assert not self.training
        x_features = self.get_features(batch)
        x_cat = self.one_to_one_catalog_net(x_features)
        est_cat = self.var_dist.sample(x_cat, use_mode=use_mode, filter_by_n_sources=True)
        est_cat = self._accumulate_stacked_tile_cat(est_cat)
        est_cat = self._choose_topk(est_cat, topk=1)
        return self._add_source_mask(est_cat)

    def _compute_loss(self, batch, logging_name):
        batch_size = batch["images"].shape[0]

        # filter out undetectable sources and split catalog by flux
        target_cat = TileCatalog(batch["tile_catalog"])
        # TODO: don't order the light sources by brightness; softmax instead
        target_cat = target_cat.get_brightest_sources_per_tile(
            top_k=self.var_dist.repeat_times,
            band=self.reference_band,
        )

        x_features = self.get_features(batch)

        one_to_one_x_cat = self.one_to_one_catalog_net(x_features.detach())
        loss1 = self.var_dist.compute_nll(one_to_one_x_cat, target_cat, topk=1)
        one_to_many_x_cat = self.one_to_many_catalog_net(x_features)
        loss2 = self.var_dist.compute_nll(one_to_many_x_cat, target_cat, topk=self.one_to_topk)
        loss = loss1 + loss2

        nan_mask = torch.isnan(loss)
        if nan_mask.any():
            loss = loss[~nan_mask]
            msg = f"NaN detected in loss. Ignored {nan_mask.sum().item()} NaN values."
            warnings.warn(msg)

        # could normalize by the number of tile predictions, rather than number of tiles
        loss = loss.sum() / loss.numel()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)

        return loss
