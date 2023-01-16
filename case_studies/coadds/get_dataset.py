#!/usr/bin/env python3
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from joblib import Parallel, delayed
from tqdm import tqdm

from case_studies.coadds.coadds import CoaddGalsimBlends


def _task(ds: CoaddGalsimBlends, n_dithers):
    outputs = {}
    full_cat, dithers = ds.sample_full_catalog()
    assert dithers.shape == (50, 2)
    for d in n_dithers:
        diths = dithers[:d]
        _, coadd, single, _ = ds.get_images(full_cat, diths)
        outputs[f"coadd_{d}"] = coadd
    outputs["single"] = single
    truth_params = {**full_cat}
    truth_params["plocs"] = full_cat.plocs
    truth_params["n_sources"] = full_cat.n_sources
    outputs["truth"] = truth_params
    return outputs


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):
    n_samples = 30000
    n_dithers = [5, 10, 25, 35, 50]
    size = 88
    prior_kwargs = {"n_dithers": max(n_dithers)}
    seed = cfg.get_data.seed
    pl.seed_everything(seed)

    ds: CoaddGalsimBlends = instantiate(cfg.datasets.galsim_blends_coadds, prior=prior_kwargs)
    output = {f"coadd_{d}": torch.zeros(n_samples, 1, size, size) for d in n_dithers}
    output["single"] = torch.zeros(n_samples, 1, size, size)
    results = Parallel(n_jobs=64)(delayed(_task)(ds, n_dithers) for _ in tqdm(range(n_samples)))
    for ii, res in enumerate(results):
        output["single"][ii] = res["single"]
        for d in n_dithers:
            output[f"coadd_{d}"][ii] = res[f"coadd_{d}"]
        for k, v in res["truth"].items():
            if output.get(k, None) is None:
                output[k] = v
            else:
                output[k] = torch.vstack([output[k], v])
    output["n_sources"] = output["n_sources"][:, 0]

    # split into train/test/val
    train = {k: v[:10000] for k, v in output.items()}
    val = {k: v[10000:20000] for k, v in output.items()}
    test = {k: v[20000:] for k, v in output.items()}

    # create all needed paths for this experiment
    output = Path(cfg.paths.output)
    dataset_dir = output.joinpath(f"datasets/{seed}")
    dataset_dir.mkdir(exist_ok=False)
    output.joinpath(f"weights/{seed}").mkdir(exist_ok=False)
    output.joinpath(f"cache/{seed}").mkdir(exist_ok=False)
    output.joinpath(f"figs/{seed}").mkdir(exist_ok=False)

    # saved datasets
    torch.save(train, dataset_dir / "train.pt")
    torch.save(val, dataset_dir / "val.pt")
    torch.save(test, dataset_dir / "test.pt")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
