import logging
from os import environ, getenv
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler
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
                file_datum: FileDatum = {k: v[0] for k, v in batch.items() if k != "tile_catalog"}
                file_datum["tile_catalog"] = {k: v[i] for k, v in batch["tile_catalog"].items()}
                file_data.append(file_datum)

        with open(f"{cached_data_path}/{gen_cfg.file_prefix}_{file_idx}.pt", "wb") as f:
            torch.save(file_data, f)


# ============================== Training mode ==============================


def train(train_cfg: DictConfig):  # pylint: disable=too-many-branches, too-many-statements
    # setup seed
    pl.seed_everything(train_cfg.seed)

    # setup dataset
    data_source_cfg = train_cfg.data_source
    dataset = instantiate(data_source_cfg)

    # setup model
    encoder = instantiate(train_cfg.encoder)

    # load pretrained weights
    if train_cfg.pretrained_weights is not None:
        enc_state_dict = torch.load(train_cfg.pretrained_weights)
        if train_cfg.pretrained_weights.endswith(".ckpt"):
            enc_state_dict = enc_state_dict["state_dict"]
        encoder.load_state_dict(enc_state_dict)

    # setup logger
    logger = False
    if train_cfg.trainer.logger:
        logger = TensorBoardLogger(
            save_dir=train_cfg.output_dir,
            name=train_cfg.name,
            version=train_cfg.version,
            default_hp_metric=False,
        )
        # no idea why calling resolve is necessary now, but it is
        OmegaConf.resolve(train_cfg)
        logger.log_hyperparams(train_cfg)

    # setup profiling
    profiler = None
    if train_cfg.trainer.profiler:
        profiler = AdvancedProfiler(filename="profile")

    callbacks = []

    # setup checkpointing
    if train_cfg.trainer.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            filename="best_encoder",
            save_top_k=1,
            verbose=True,
            monitor="val/_loss",
            mode="min",
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

    # setup early stopping
    if train_cfg.enable_early_stopping:
        early_stopping = EarlyStopping(monitor="val/_loss", mode="min", patience=train_cfg.patience)
        callbacks.append(early_stopping)

    trainer = instantiate(train_cfg.trainer, logger=logger, profiler=profiler, callbacks=callbacks)

    # train!
    trainer.fit(encoder, datamodule=dataset)

    # test!
    if train_cfg.testing:
        trainer.test(encoder, datamodule=dataset)


# ============================== Prediction mode ==============================


def predict(predict_cfg):
    encoder = instantiate(predict_cfg.encoder)
    enc_state_dict = torch.load(predict_cfg.weight_save_path)
    if predict_cfg.weight_save_path.endswith(".ckpt"):
        enc_state_dict = enc_state_dict["state_dict"]
    encoder.load_state_dict(enc_state_dict)
    dataset = instantiate(predict_cfg.dataset)
    trainer = instantiate(predict_cfg.trainer)
    enc_output = trainer.predict(encoder, datamodule=dataset)

    mode_cats = [b["mode_cat"].to_full_catalog() for b in enc_output]
    return dict(zip(dataset.image_ids(), mode_cats))


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
    main()  # pylint: disable=no-value-for-parameter

# pragma: no cover
