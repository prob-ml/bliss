import torch
from hydra.utils import instantiate


class TestEncoder:
    def test_encode_multi_source_catalog(self, cfg, multi_source_dataloader, encoder):
        batch = next(iter(multi_source_dataloader))
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(cfg.predict.device)
        pred = encoder.encode_batch(batch)
        encoder.variational_mode(pred)

    def test_encode_with_psf(self, cfg, multiband_dataloader):
        batch = next(iter(multiband_dataloader))
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(cfg.predict.device)

        encoder_params = {
            "bands": [2],
            "input_transform_params": {"use_deconv_channel": True, "concat_psf_params": True},
        }
        encoder = instantiate(cfg.encoder, **encoder_params).to(cfg.predict.device)
        pred = encoder.encode_batch(batch)
        encoder.variational_mode(pred)
