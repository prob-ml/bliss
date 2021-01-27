import torch
from pathlib import Path


def test_galaxy_vae(galaxy_vae_setup, paths, devices):
    use_cuda = devices.use_cuda
    device = devices.device
    overrides = {
        "model": "galaxy_net",
        "dataset": "saved_catsim",
        "training": "unittest",
        "training.n_epochs": 101 if use_cuda else 2,
        "training.trainer.limit_train_batches": 20 if use_cuda else 2,
        "training.trainer.limit_val_batches": 1,
        "training.trainer.check_val_every_n_epoch": 50 if use_cuda else 1,
    }

    # prepare galaxy
    one_galaxy = torch.load(Path(paths["data"]).joinpath("1_catsim_galaxy.pt"))
    galaxy = one_galaxy["images"].to(device)
    background = one_galaxy["background"].to(device)

    # train galaxy_vae
    galaxy_vae = galaxy_vae_setup.get_trained_vae(overrides)

    with torch.no_grad():
        galaxy_vae.eval()
        pred_image, _, _ = galaxy_vae(galaxy, background)

    residual = (galaxy - pred_image) / torch.sqrt(galaxy)

    # only expect tests to pass in cuda:
    if not devices.use_cuda:
        return

    # check residuals follow gaussian noise, most pixels are between 68%
    n_pixels = residual.size(-1) ** 2
    assert (residual.abs() <= 1).sum() >= n_pixels * 0.5
