import logging
import random
from os import environ, getenv
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bliss.simulator.simulated_dataset import FileDatum

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
        file_data: List[FileDatum] = []

        for _ in tqdm(range(gen_cfg.n_batches_per_file), desc="Generating one dataset file"):
            batch = simulated_dataset.get_batch()

            # flatten batches
            for i in range(gen_cfg.simulator.prior.batch_size):
                file_datum: FileDatum = {k: v[i] for k, v in batch.items() if k != "tile_catalog"}
                file_datum["tile_catalog"] = {k: v[i] for k, v in batch["tile_catalog"].items()}
                file_data.append(file_datum)

        with open(f"{cached_data_path}/{gen_cfg.file_prefix}_{file_idx}.pt", "wb") as f:
            torch.save(file_data, f)


# ============================== Training mode ==============================


def train(train_cfg: DictConfig):
    # setup seed
    if train_cfg.seed == "random":
        pl.seed_everything(random.randint(1e4, 1e5 - 1))
    else:
        pl.seed_everything(train_cfg.seed)

    # setup dataset, encoder, and trainer
    dataset = instantiate(train_cfg.data_source)
    encoder = instantiate(train_cfg.encoder)
    trainer = instantiate(train_cfg.trainer)

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
    trainer.fit(encoder, datamodule=dataset)

    # load best model for test
    if train_cfg.test_best:
        best_model_path = trainer.callbacks["checkpointing"].best_model_path
        enc_state_dict = torch.load(best_model_path)
        enc_state_dict = enc_state_dict["state_dict"]
        encoder.load_state_dict(enc_state_dict)

    # test!
    if train_cfg.testing:
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
    if not getenv("BLISS_HOME"):
        project_path = Path(__file__).resolve()
        bliss_home = project_path.parents[1]
        environ["BLISS_HOME"] = bliss_home.as_posix()

        logger = logging.getLogger(__name__)
        logger.warning(
            "WARNING: BLISS_HOME not set, setting to project root %s\n",  # noqa: WPS323
            environ["BLISS_HOME"],
        )

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
