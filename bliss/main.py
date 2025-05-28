import logging
from multiprocessing import current_process, get_context
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from bliss.catalog import TileCatalog
from bliss.global_env import GlobalEnv

# ============================== Data Generation ==============================


def generate(gen_cfg: DictConfig):
    # create cached_data_path if it doesn't exist
    cached_data_path = Path(gen_cfg.cached_data_path)
    if not cached_data_path.exists():
        cached_data_path.mkdir(parents=True)
    logger = logging.getLogger(__name__)
    logger.info(f"Data will be saved to {cached_data_path}")

    # log the Hydra config used to generate data to cached_data_path
    with open(cached_data_path / "hparams.yaml", "w", encoding="utf-8") as f:
        OmegaConf.resolve(gen_cfg)
        OmegaConf.save(gen_cfg, f)

    ctx = get_context("fork")
    args = ((gen_cfg, i) for i in range(gen_cfg.n_image_files))
    with ctx.Pool(processes=gen_cfg.n_processes) as pool:
        pool.starmap(generate_one_file, args)


def generate_one_file(gen_cfg: DictConfig, file_idx: int):
    pl.seed_everything(gen_cfg.seed + file_idx)
    prior = instantiate(gen_cfg.prior)
    decoder = instantiate(gen_cfg.decoder)

    file_data = []

    logger = logging.getLogger(__name__)

    for b in range(gen_cfg.n_batches_per_file):
        logger.info(f"{current_process().name}: Generating batch {b} of file {file_idx}")

        tile_catalog = prior.sample()
        images, psf_params = decoder.render_images(tile_catalog)

        ttc = gen_cfg.tiles_to_crop
        tile_catalog = tile_catalog.symmetric_crop(ttc)
        ptc = ttc * decoder.tile_slen  # pixels to crop
        if ptc > 0:
            images = images[:, :, ptc:-ptc, ptc:-ptc]

        batch = {
            "tile_catalog": tile_catalog,
            "images": images,
            "psf_params": psf_params,
        }

        if gen_cfg.store_full_catalog:
            tile_cat = TileCatalog(batch["tile_catalog"])
            full_cat = tile_cat.to_full_catalog(gen_cfg.decoder.tile_slen)

        # flatten batches
        for i in range(gen_cfg.prior.batch_size):
            file_datum = {k: v[i] for k, v in batch.items() if k != "tile_catalog"}

            if not gen_cfg.store_full_catalog:
                file_datum["tile_catalog"] = {k: v[i] for k, v in batch["tile_catalog"].items()}
            else:
                file_datum["full_catalog"] = {k: v[i] for k, v in full_cat.items()}

            file_data.append(file_datum)

    filename = f"dataset_{file_idx}_size_{len(file_data)}.pt"
    with open(Path(gen_cfg.cached_data_path) / filename, "wb") as f:
        torch.save(file_data, f)


# ============================== Training mode ==============================


def train(train_cfg: DictConfig):
    # setup seed
    seed = pl.seed_everything(train_cfg.seed)
    GlobalEnv.seed_in_this_program = seed

    if train_cfg.matmul_precision:
        torch.set_float32_matmul_precision(train_cfg.matmul_precision)

    # setup dataset, encoder, callbacks and trainer
    dataset = instantiate(train_cfg.data_source)
    encoder = instantiate(train_cfg.encoder)
    callbacks = instantiate(train_cfg.callbacks)
    trainer = instantiate(train_cfg.trainer, callbacks=list(callbacks.values()))

    # load pretrained weights
    if train_cfg.pretrained_weights is not None:
        enc_state_dict = torch.load(train_cfg.pretrained_weights)
        if train_cfg.pretrained_weights.endswith(".ckpt"):
            enc_state_dict = enc_state_dict["state_dict"]
        encoder.load_state_dict(enc_state_dict)

    # log the training config
    if trainer.logger:
        OmegaConf.resolve(train_cfg)
        trainer.logger.log_hyperparams(train_cfg)

    # train!
    trainer.fit(encoder, datamodule=dataset, ckpt_path=train_cfg.ckpt_path)

    # test!
    # load best model for test
    best_model_path = callbacks["checkpointing"].best_model_path
    enc_state_dict = torch.load(best_model_path)
    enc_state_dict = enc_state_dict["state_dict"]
    encoder.load_state_dict(enc_state_dict)
    trainer.test(encoder, datamodule=dataset)


# ============================== Prediction mode ==============================


def predict(predict_cfg):
    encoder = instantiate(predict_cfg.encoder)
    enc_state_dict = torch.load(predict_cfg.weight_save_path, map_location=predict_cfg.device)
    if predict_cfg.weight_save_path.endswith(".ckpt"):
        enc_state_dict = enc_state_dict["state_dict"]
    encoder.load_state_dict(enc_state_dict)
    dataset = instantiate(predict_cfg.dataset)
    trainer = instantiate(predict_cfg.trainer)
    enc_output = trainer.predict(encoder, datamodule=dataset)

    return dict(zip(dataset.image_ids(), enc_output))


# pragma: no cover
# ============================== CLI ==============================


# config_path should be overriden when running `bliss` poetry executable
# e.g., `bliss -cp case_studies/summer_template -cn config`
@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def main(cfg):
    """Main entry point(s) for BLISS."""
    if cfg.mode == "generate":
        generate(cfg.generate)
    elif cfg.mode == "train":
        train(cfg.train)
    elif cfg.mode == "predict":
        predict(cfg.predict)
    else:
        raise KeyError


if __name__ == "__main__":
    main()

# pragma: no cover
