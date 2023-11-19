import torch
from hydra.utils import instantiate


def predict(predict_cfg):
    encoder = instantiate(predict_cfg.encoder)
    encoder.load_state_dict(torch.load(predict_cfg.weight_save_path))
    dataset = instantiate(predict_cfg.dataset)
    trainer = instantiate(predict_cfg.trainer)
    enc_output = trainer.predict(encoder, datamodule=dataset)

    est_cats = [b["est_cat"].to_full_catalog() for b in enc_output]
    return dict(zip(dataset.image_ids(), est_cats))
