import torch
from hydra.utils import instantiate


def predict(cfg):
    encoder = instantiate(cfg.encoder)
    encoder.load_state_dict(torch.load(cfg.predict.weight_save_path))
    dataset = instantiate(cfg.predict.dataset)
    trainer = instantiate(cfg.training.trainer)
    enc_output = trainer.predict(encoder, datamodule=dataset)

    est_cats = [b["est_cat"].to_full_catalog() for b in enc_output]
    return dict(zip(dataset.image_ids(), est_cats))
