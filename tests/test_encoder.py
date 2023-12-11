import torch
from hydra.utils import instantiate


class TestEncoder:
    def test_encode_multi_source_catalog(self, cfg, multi_source_dataloader):
        batch = next(iter(multi_source_dataloader))
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(cfg.predict.device)
        encoder = instantiate(cfg.encoder).to(cfg.predict.device)
        encoder.sample(batch, use_mode=True)

    def test_encode_with_psf(self, cfg, multiband_dataloader):
        batch = next(iter(multiband_dataloader))
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(cfg.predict.device)

        encoder_params = {
            "image_normalizer": {
                "use_deconv_channel": True,
                "concat_psf_params": True,
                "bands": [2],
            },
        }
        encoder = instantiate(cfg.encoder, **encoder_params).to(cfg.predict.device)
        encoder.sample(batch, use_mode=True)
        encoder.sample(batch, use_mode=False)
