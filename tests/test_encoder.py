import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader


class TestEncoder:
    def test_sample_multisource_catalog(self, cfg):
        with open(f"{cfg.paths.test_data}/test_multi_source.pt", "rb") as f:
            data = torch.load(f)
        multi_source_dataloader = DataLoader(data, batch_size=8, shuffle=False)

        batch = next(iter(multi_source_dataloader))
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(cfg.predict.device)
        encoder = instantiate(cfg.encoder).to(cfg.predict.device)
        encoder.sample(batch, use_mode=True)

    def test_sample_with_psf(self, cfg):
        with open(f"{cfg.paths.test_data}/multiband_data/dataset_0.pt", "rb") as f:
            data = torch.load(f)
        multiband_dataloader = DataLoader(data, batch_size=8, shuffle=False)

        batch = next(iter(multiband_dataloader))
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(cfg.predict.device)

        encoder = instantiate(cfg.encoder).to(cfg.predict.device)
        encoder.sample(batch, use_mode=True)
        encoder.sample(batch, use_mode=False)
