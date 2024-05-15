#!/usr/bin/env python3

from pathlib import Path

import click
import pytorch_lightning as pl
import torch
from astropy.table import Table

from bliss.datasets.galsim_blends import generate_dataset
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor
from bliss.reporting import get_snr

HOME_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"
cat = Table.read(HOME_DIR / "data/OneDegSq.fits")
CATSIM_TABLE = cat[cat["i_ab"] < 27.3]
star_mags = column_to_tensor(Table.read(HOME_DIR / "data/stars_med_june2018.fits"), "i_ab")
STAR_MAGS = star_mags[star_mags > 20]
PSF = get_default_lsst_psf()


@click.command()
@click.option("--n-samples", default=10000, type=int)
@click.option("-s", "--seed", default=1, type=int)
@click.option("--mode", type=str, required=True)
@click.option("-o", "--overwrite", is_flag=True, default=False)
def main(n_samples: int, seed: int, mode: str, overwrite: bool):
    assert mode in {"single", "blend"}

    pl.seed_everything(seed)

    if mode == "single":
        dataset_file = DATA_DIR / "single_galaxies_test.pt"
        if not overwrite and Path(dataset_file).exists():
            raise IOError("File already exists and overwrite flag is 'False'.")

        dataset = generate_dataset(
            n_samples,
            CATSIM_TABLE,
            STAR_MAGS,
            psf=PSF,
            max_n_sources=1,
            galaxy_density=1000,
            star_density=0,
            slen=53,
            bp=0,
            max_shift=0,
            add_galaxies_in_padding=False,
        )

        # compute SNR
        dataset["snr"] = get_snr(dataset["noiseless"], dataset["background"])

        # convert everything to float and remove useless params
        params_to_remove = {
            "individuals",
            "galaxy_params",
            "star_fluxes",
            "plocs",
            "n_sources",
            "fluxes",
            "star_bools",
            "galaxy_bools",
        }
        for p in params_to_remove:
            dataset.pop(p)

        for p1, q in dataset.items():
            dataset[p1] = q.float()

        torch.save(dataset, dataset_file)

    else:
        dataset_file = DATA_DIR / "blends_test.pt"
        if not overwrite and Path(dataset_file).exists():
            raise IOError("File already exists and overwrite flag is 'False'.")

        dataset = generate_dataset(
            n_samples,
            CATSIM_TABLE,
            STAR_MAGS,
            psf=PSF,
            max_n_sources=15,
        )

        torch.save(dataset, dataset_file)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
