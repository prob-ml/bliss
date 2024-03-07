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


def _save_dataset(ds, train_path, val_path, test_path, n_samples: int, overwrite=False, njobs=1):
    assert n_samples % 3 == 0
    tpath, vpath, ttpath = Path(train_path), Path(val_path), Path(test_path)

    if tpath.exists() or vpath.exists() or ttpath.exists():
        if not overwrite:
            raise ValueError("Overwrite turned on, but files exists.")

    results = Parallel(n_jobs=njobs)(delayed(_task)(ds, ii) for ii in tqdm(range(n_samples)))
    output = torch.cat(results)
    assert output.shape[0] == n_samples

    # nn need all input tensors to be float32
    if "images" in output.keys() and "background" in output.keys():
        output["images"] = output["images"].float()
        output["background"] = output["background"].float()

    div1, div2 = n_samples // 3, n_samples // 3 * 2
    torch.save(output[:div1], tpath)
    torch.save(output[div1:div2], vpath)
    torch.save(output[div2:], ttpath)


def _generate_single_galaxy_datasets(cfg, n_samples, overwrite):
    ds = instantiate(cfg.single_galaxy_datasets.single_galaxies)
    train_path = Path(cfg.single_galaxy_datasets.train_saved_single_galaxies.dataset_file)
    val_path = Path(cfg.single_galaxy_datasets.val_saved_single_galaxies.dataset_file)
    test_path = Path(cfg.plots.test_datasets_files.single_galaxies)
    _save_dataset(ds, train_path, val_path, test_path, n_samples, overwrite, njobs=1)


def _generate_blends_datasets(cfg, n_samples, overwrite):
    ds = instantiate(cfg.blends_datasets.blends)
    train_path = Path(cfg.blends_datasets.train_saved_blends.dataset_file)
    val_path = Path(cfg.blends_datasets.val_saved_blends.dataset_file)
    test_path = Path(cfg.plots.test_datasets_files.blends)
    _save_dataset(ds, train_path, val_path, test_path, n_samples, overwrite, njobs=1)


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
