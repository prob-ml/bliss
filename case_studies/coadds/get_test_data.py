#!/usr/bin/env python3
import hydra
import torch
from hydra.utils import instantiate
from joblib import Parallel, delayed
from tqdm import tqdm

from case_studies.coadds.coadd_decoder import CoaddGalsimBlends


def task(ds: CoaddGalsimBlends, n_dithers):
    outputs = {}
    full_cat, dithers = ds._sample_full_catalog()
    assert dithers.shape == (50, 2)
    for d in n_dithers:
        diths = dithers[:d]
        _, coadd, single, _ = ds._get_images(full_cat, diths)
        outputs[f"coadd_{d}"] = coadd
    outputs["single"] = single
    truth_params = {**full_cat}
    truth_params["plocs"] = full_cat.plocs
    truth_params["n_sources"] = full_cat.n_sources
    outputs["truth"] = truth_params
    return outputs


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    n_samples = 10000
    n_dithers = [5, 10, 25, 35, 50]
    size = 88
    ds: CoaddGalsimBlends = instantiate(cfg.datasets.galsim_blends_coadds, prior={"n_dithers": 50})
    output = {f"coadd_{d}": torch.zeros(n_samples, 1, size, size) for d in n_dithers}
    output["single"] = torch.zeros(n_samples, 1, size, size)
    results = Parallel(n_jobs=20)(delayed(task)(ds, n_dithers) for _ in tqdm(range(n_samples)))
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

    torch.save(output, "output/test_dataset_poisson.pt")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
