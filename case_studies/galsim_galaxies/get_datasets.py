#!/usr/bin/env python3

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from joblib import Parallel, delayed
from tqdm import tqdm


def _task(ds, idx):
    return ds[idx]


def _generate_single_galaxy_datasets(cfg, n_samples, overwrite):
    assert n_samples == 30000
    ds = instantiate(cfg.single_galaxy_datasets.single_galaxies)

    train_path = Path(cfg.single_galaxy_datasets.train_saved_single_galaxies.dataset_file)
    val_path = Path(cfg.single_galaxy_datasets.val_saved_single_galaxies.dataset_file)
    test_path = Path(cfg.plots.test_datasets.single_galaxies_test_file)

    if train_path.exists() or val_path.exists() or test_path.exists():
        if not overwrite:
            raise ValueError("Overwrite turned on, but files exists.")

    results = Parallel(n_jobs=1)(delayed(_task)(ds, ii) for ii in tqdm(range(n_samples)))
    output = torch.cat(results)
    assert output.shape[0] == n_samples

    torch.save(output[:10000], train_path)
    torch.save(output[10000:20000], val_path)
    torch.save(output[20000:30000], test_path)


def _generate_blends_datasets(cfg, n_samples, overwrite):
    ds = instantiate(cfg.blends_datasets.blends)
    train_path = Path(cfg.blends_datasets.train_saved_blends.dataset_file)
    val_path = Path(cfg.blends_datasets.val_saved_blends.dataset_file)
    test_path = Path(cfg.plots.test_datasets.blends_test_file)

    if train_path.exists() or val_path.exists() or test_path.exists():
        if not overwrite:
            raise ValueError("Overwrite turned on, but files exists.")

    results = Parallel(n_jobs=1)(delayed(_task)(ds, ii) for ii in tqdm(range(n_samples)))
    output = torch.cat(results)
    assert output.shape[0] == n_samples

    torch.save(output[:10000], train_path)
    torch.save(output[10000:20000], val_path)
    torch.save(output[20000:30000], test_path)


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):

    # setup
    n_samples = cfg.get_data.n_samples
    seed = cfg.get_data.seed
    ds_name = cfg.get_data.ds_name
    overwrite = cfg.get_data.overwrite
    pl.seed_everything(seed)

    if ds_name == "single_gal":
        _generate_single_galaxy_datasets(cfg, n_samples, overwrite)

    elif ds_name == "blends":
        _generate_blends_datasets(cfg, n_samples, overwrite)

    else:
        raise ValueError("Dataset not found in config.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
