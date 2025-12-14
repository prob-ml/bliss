import pytest
import torch
from hydra.utils import instantiate

from bliss.global_env import GlobalEnv
from bliss.main import train


class TestDC2:
    @pytest.mark.run_first
    def test_dc2_size_and_type(self, cfg, monkeypatch):
        dc2 = instantiate(cfg.surveys.dc2)
        dc2.prepare_data()
        dc2.setup(stage="fit")

        monkeypatch.setattr(GlobalEnv, "seed_in_this_program", 0)
        monkeypatch.setattr(GlobalEnv, "current_encoder_epoch", 0)

        dc2 = list(dc2.train_dataloader())

        assert dc2[0]["images"].shape[1] == 6
        assert len(dc2) == 7  # n_image_split=3 → 9 tiles, 80% train split → 7 batches

        params = (
            "locs",
            "n_sources",
            "source_type",
            "fluxes",
        )

        for k in params:
            assert isinstance(dc2[0]["tile_catalog"][k], torch.Tensor)

        for k in ("images", "psf_params"):
            assert isinstance(dc2[0][k], torch.Tensor)

    @pytest.mark.run_first
    def test_train_on_dc2(self, cfg):
        cfg.encoder.survey_bands = ["u", "g", "r", "i", "z", "y"]
        cfg.train.data_source = cfg.surveys.dc2
        cfg.train.pretrained_weights = None
        cfg.encoder.image_normalizers.psf.num_psf_params = 4

        cfg.encoder.var_dist.factors = [
            {
                "_target_": "bliss.encoder.variational_dist.BernoulliFactor",
                "name": "n_sources",
                "sample_rearrange": None,
                "nll_rearrange": None,
                "nll_gating": None,
            },
            {
                "_target_": "bliss.encoder.variational_dist.TDBNFactor",
                "name": "locs",
                "sample_rearrange": "b ht wt d -> b ht wt 1 d",
                "nll_rearrange": "b ht wt 1 d -> b ht wt d",
                "nll_gating": {"_target_": "bliss.encoder.variational_dist.SourcesGating"},
            },
            {
                "_target_": "bliss.encoder.variational_dist.BernoulliFactor",
                "name": "source_type",
                "sample_rearrange": "b ht wt -> b ht wt 1 1",
                "nll_rearrange": "b ht wt 1 1 -> b ht wt",
                "nll_gating": {"_target_": "bliss.encoder.variational_dist.SourcesGating"},
            },
            {
                "_target_": "bliss.encoder.variational_dist.LogNormalFactor",
                "name": "fluxes",
                "dim": 6,
                "sample_rearrange": "b ht wt d -> b ht wt 1 d",
                "nll_rearrange": "b ht wt 1 d -> b ht wt d",
                "nll_gating": {"_target_": "bliss.encoder.variational_dist.SourcesGating"},
            },
        ]

        for f in cfg.variational_factors:
            if f.name == "fluxes":
                f.dim = 6

        train(cfg.train)
