from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bliss.catalog import TileCatalog
from bliss.global_env import GlobalEnv

# ============================== Data Generation ==============================


def generate(gen_cfg: DictConfig):
    # it's more efficient to launch multiple independent processes than to use workers
    simulated_dataset = instantiate(gen_cfg.simulator, num_workers=0)

    # create cached_data_path if it doesn't exist
    cached_data_path = Path(gen_cfg.cached_data_path)
    if not cached_data_path.exists():
        cached_data_path.mkdir(parents=True)
    print("Data will be saved to {}".format(cached_data_path))  # noqa: WPS421

    # log the Hydra config used to generate data to cached_data_path
    with open(cached_data_path / "hparams.yaml", "w", encoding="utf-8") as f:
        OmegaConf.resolve(gen_cfg)
        OmegaConf.save(gen_cfg, f)

    # n_image_files is technically "n_image_files for this process"
    process_index = gen_cfg.get("process_index", 0)
    files_start_idx = process_index * gen_cfg.n_image_files

    # overwrites any existing cached image files
    file_idxs = range(files_start_idx, files_start_idx + gen_cfg.n_image_files)
    for file_idx in tqdm(file_idxs, desc="Generating and writing dataset files"):
        file_data = []

        for _ in tqdm(range(gen_cfg.n_batches_per_file), desc="Generating one dataset file"):
            batch = simulated_dataset.get_batch()

            if gen_cfg.store_full_catalog:
                tile_cat = TileCatalog(gen_cfg.simulator.prior.tile_slen, batch["tile_catalog"])
                full_cat = tile_cat.to_full_catalog()

            # flatten batches
            for i in range(gen_cfg.simulator.prior.batch_size):
                file_datum = {k: v[i] for k, v in batch.items() if k != "tile_catalog"}

                if not gen_cfg.store_full_catalog:
                    file_datum["tile_catalog"] = {k: v[i] for k, v in batch["tile_catalog"].items()}
                else:
                    file_datum["full_catalog"] = {k: v[i] for k, v in full_cat.items()}

                file_data.append(file_datum)

        filename = f"{gen_cfg.file_prefix}_{file_idx}_size_{len(file_data)}.pt"
        with open(cached_data_path / filename, "wb") as f:
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
