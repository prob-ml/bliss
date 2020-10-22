import torch
from bliss import wake
import pytorch_lightning as pl


def test_star_wake(get_dataset, get_trained_encoder, paths, devices):

    device = devices.device
    overrides = dict(model="basic_sleep_star", training="cpu", dataset="cpu")
    dataset = get_dataset(overrides)
    trained_encoder = get_trained_encoder(dataset, overrides)

    # load the test image
    # 3-stars 30*30 pixels.
    test_path = paths["data"].joinpath("3_star_test.pt")
    test_star = torch.load(test_path, map_location="cpu")
    test_image = test_star["images"].to(device)
    test_slen = test_image.size(-1)
    image_decoder = dataset.image_decoder.to(device)
    background_value = image_decoder.background.mean().item()

    # initialize background params, which will create the true background
    init_background_params = torch.zeros(1, 3, device=device)
    init_background_params[0, 0] = background_value

    n_samples = 1
    hparams = {"n_samples": n_samples, "lr": 0.001}
    assert image_decoder.slen == test_slen
    wake_phase_model = wake.WakeNet(
        trained_encoder,
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
