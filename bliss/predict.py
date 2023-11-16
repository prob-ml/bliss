import torch
from hydra.utils import instantiate


def predict(cfg):
    encoder = instantiate(cfg.encoder)
    encoder.eval()
    encoder.load_state_dict(torch.load(cfg.predict.weight_save_path))

    dataset = instantiate(cfg.predict.dataset)

    trainer = instantiate(cfg.training.trainer)
    enc_output = trainer.predict(encoder, datamodule=dataset)

    est_cat_tables = []
    for batch_output in enc_output:
        est_cat = batch_output["est_cat"]
        full_cat = est_cat.to_full_catalog()
        astropy_cat = full_cat.to_astropy_table(cfg.encoder.survey_bands)
        est_cat_tables.append(astropy_cat)

    # eventually we should return variational distribution samples too
    return est_cat_tables
