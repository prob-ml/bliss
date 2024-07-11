# flake8: noqa
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch import distributed


def save_tile_result(tile_result, tile_idx, gpu_rank, save_path):
    save_path = Path(save_path) / f"tile_{tile_idx}_gpu_{gpu_rank}.pt"
    torch.save(tile_result, save_path)


def load_encoder(predict_cfg):
    encoder = instantiate(predict_cfg.encoder)
    enc_state_dict = torch.load(predict_cfg.weight_save_path)
    enc_state_dict = enc_state_dict["state_dict"]
    encoder.load_state_dict(enc_state_dict)
    return encoder


def inference(predict_cfg):
    encoder = load_encoder(predict_cfg)
    trainer = instantiate(predict_cfg.trainer)
    dataset = instantiate(predict_cfg.cached_dataset)
    enc_output = trainer.predict(encoder, datamodule=dataset)
    gpu_rank = (
        distributed.get_rank() if distributed.is_available() and distributed.is_initialized() else 0
    )
    for tile_idx, batch in enumerate(enc_output):
        save_tile_result(batch, tile_idx, gpu_rank, predict_cfg.output_save_path)


def main():
    with initialize(config_path=".", version_base=None):
        cfg = compose("config")

    predict_cfg = cfg.predict
    inference(predict_cfg=predict_cfg)


if __name__ == "__main__":
    main()
