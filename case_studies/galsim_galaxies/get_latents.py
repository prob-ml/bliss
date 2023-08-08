#!/usr/bin/env python3
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate

from bliss.models.galaxy_net import OneCenteredGalaxyAE

pl.seed_everything(41)


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):
    # load AE
    device = torch.device("cuda:0")

    autoencoder: OneCenteredGalaxyAE = instantiate(cfg.models.galaxy_net)
    autoencoder.load_state_dict(torch.load(cfg.models.prior.galaxy_prior.autoencoder_ckpt))
    autoencoder = autoencoder.to(device).eval()

    # create galaxies to encode
    latents_file = Path(cfg.models.prior.galaxy_prior.latents_file)

    # delete file if it exists using
    if latents_file.exists():
        os.remove(latents_file)
        print("WARNING: Latent file already exists. Overwriting.")

    ds = instantiate(cfg.datasets.sdss_galaxies, batch_size=512, n_batches=20, num_workers=20)
    dl = ds.train_dataloader()
    latents = autoencoder.generate_latents(dl, n_batches=20)
    latents = latents.detach().to("cpu")
    torch.save(latents, latents_file)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
