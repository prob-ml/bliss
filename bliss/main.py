import logging
from os import environ, getenv
from pathlib import Path
from typing import Dict, List

import hydra
import pytorch_lightning as pl
import torch
from einops import rearrange
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
    max_images_per_file = gen_cfg.max_images_per_file
    cached_data_path = gen_cfg.cached_data_path
    n_workers_per_process = gen_cfg.n_workers_per_process

    # largest `batch_size` multiple <= `max_images_per_file`
    bs = gen_cfg.batch_size
    images_per_file = (max_images_per_file // bs) * bs
    assert images_per_file >= bs, "max_images_per_file too small"

    # number of files needed to store >= `n_batches` * `batch_size` images
    # in <= `images_per_file`-image files
    n_files = -(gen_cfg.n_batches * bs // -images_per_file)  # ceil division

    # note: this is technically "n_files for this process"
    process_index = gen_cfg.get("process_index", 0)
    files_start_idx = process_index * n_files

    # use SimulatedDataset to generate data in minibatches iteratively,
    # then concatenate before caching to disk via pickle
    simulator = instantiate(
        gen_cfg.simulator,
        num_workers=n_workers_per_process,
        prior={"batch_size": bs},
    )
    simulated_dataset = simulator.train_dataloader().dataset

    # create cached_data_path if it doesn't exist
    if not Path(cached_data_path).exists():
        Path(cached_data_path).mkdir(parents=True)
    print("Data will be saved to {}".format(cached_data_path))  # noqa: WPS421

    # Save Hydra config (used to generate data) to cached_data_path
    with open(f"{gen_cfg.cached_data_path}/hparams.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(gen_cfg, f)

    # assume overwriting any existing cached image files
    file_idxs = range(files_start_idx, files_start_idx + n_files)
    for file_idx in tqdm(file_idxs, desc="Generating and writing cached dataset files"):
        batch_data = _generate_data(
            images_per_file // bs, simulated_dataset, "Simulating images in batches for file"
        )
        file_data = _itemize_data(batch_data)
        with open(f"{cached_data_path}/{gen_cfg.file_prefix}_{file_idx}.pt", "wb") as f:
            torch.save(file_data, f)


def _generate_data(n_batches: int, simulated_dataset, desc="Generating data"):
    batch_data: List[Dict[str, torch.Tensor]] = []
    for _ in tqdm(range(n_batches), desc=desc):
        batch_data.append(next(iter(simulated_dataset)))
    return batch_data


def _itemize_data(batch_data) -> List[FileDatum]:
    flat_data = {}

    # flatten tile catalog
    tile_catalog_flattened = {}
    for key in batch_data[0]["tile_catalog"].keys():
        batch_tc_key = torch.stack([data["tile_catalog"][key] for data in batch_data])
        tile_catalog_flattened[key] = rearrange(batch_tc_key, "b c ... -> (b c) ...")
    flat_data["tile_catalog"] = tile_catalog_flattened

    # flatten the rest of the data
    keys = ["images", "background", "deconvolution", "psf_params"]
    for key in keys:
        if key in batch_data[0]:
            batch_ch = torch.stack([data[key] for data in batch_data])
            flat_data[key] = rearrange(batch_ch, "b c ... -> (b c) ...")

    # reconstruct data as list of single-input FileDatum dictionaries
    n_items = len(flat_data["images"])
    file_data: List[FileDatum] = []
    for i in range(n_items):
        file_datum: FileDatum = {}
        # construct a TileCatalog dictionary of ith-input tensors
        file_datum["tile_catalog"] = {
            k: flat_data["tile_catalog"][k][i] for k in flat_data["tile_catalog"].keys()
        }
        file_datum["images"] = flat_data["images"][i]
        file_datum["background"] = flat_data["background"][i]
        file_datum["deconvolution"] = flat_data["deconvolution"][i]
        file_datum["psf_params"] = flat_data["psf_params"][i]
        file_data.append(file_datum)

    return file_data


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
            save_top_k=train_cfg.save_top_k,
            save_weights_only=True,
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
    encoder.load_state_dict(torch.load(predict_cfg.weight_save_path))
    dataset = instantiate(predict_cfg.dataset)
    trainer = instantiate(predict_cfg.trainer)
    enc_output = trainer.predict(encoder, datamodule=dataset)

    est_cats = [b["est_cat"].to_full_catalog() for b in enc_output]
    return dict(zip(dataset.image_ids(), est_cats))


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
