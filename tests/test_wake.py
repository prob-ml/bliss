import torch
from bliss import wake
import pytorch_lightning as pl


def test_star_wake(sleep_setup, paths, devices):
    device = devices.device
    overrides = dict(model="sleep_star_one_tile", training="cpu", dataset="cpu")
    sleep_net = sleep_setup.get_trained_sleep(overrides)

    # load the test image
    test_path = paths["data"].joinpath("star_wake_test1.pt")
    test_star = torch.load(test_path, map_location="cpu")
    test_image = test_star["images"][0].unsqueeze(0).to(device)
    test_slen = test_star["slen"].item()
    image_decoder = sleep_net.image_decoder.to(device)
    background_value = image_decoder.background.mean().item()

    # initialize background params, which will create the true background
    init_background_params = torch.zeros(1, 3, device=device)
    init_background_params[0, 0] = background_value

    n_samples = 1
    hparams = {"n_samples": n_samples, "lr": 0.001}
    assert image_decoder.slen == test_slen
    wake_phase_model = wake.WakeNet(
        sleep_net.image_encoder,
        image_decoder,
        test_image,
        init_background_params,
        hparams,
    )

    # run the wake-phase training
    n_epochs = 1

    wake_trainer = pl.Trainer(
        gpus=devices.gpus,
        profiler=None,
        logger=False,
        checkpoint_callback=False,
        min_epochs=n_epochs,
        max_epochs=n_epochs,
        reload_dataloaders_every_epoch=True,
    )

    wake_trainer.fit(wake_phase_model)
