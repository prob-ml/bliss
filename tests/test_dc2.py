import torch
from hydra.utils import instantiate


class TestDC2:
    def test_dc2(self, cfg):
        dataset = instantiate(cfg.surveys.dc2)
        dataset.prepare_data()
        dc2_obj = dataset.dc2_data[0]

        dc2_tile = dc2_obj['tile_catalog']
        params = ("locs",
                    "n_sources",
                    "source_type",
                    "galaxy_fluxes",
                    "galaxy_params",
                    "star_fluxes",
                    "star_log_fluxes")

        for k in params:
            assert isinstance(dc2_tile[k], torch.Tensor)

        for i in ("images", "background"):
            assert isinstance(dc2_obj[i], torch.Tensor)
