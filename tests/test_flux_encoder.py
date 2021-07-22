import torch
import pytorch_lightning as pl

from hydra import initialize, compose

from bliss.models import flux_net
from bliss.datasets import simulated


class TestFluxEncoder:
    def test_flux_encoder(self, devices):

        device = devices.device

        # get parameters
        overrides = {"model": "sleep_sdss_measure_simple", "dataset": "default"}
        overrides = [f"{key}={value}" for key, value in overrides.items()]
        with initialize(config_path="./../config"):
            cfg = compose("config", overrides=overrides)

        # set device
        cfg.dataset.kwargs.update({"generate_device": str(device) if devices.use_cuda else "cpu"})
        cfg.dataset.kwargs.update({"generate_device": str(device) if devices.use_cuda else "cpu"})

        if not devices.use_cuda:
            cfg.training.update({"n_epochs": 3})
        else:
            cfg.training.update({"n_epochs": 20})

        # get dataset
        dataset = simulated.SimulatedDataset(**cfg.dataset.kwargs)

        # get encoder
        flux_estimator = flux_net.FluxEstimator(
            cfg.model.kwargs.decoder_kwargs, optimizer_params=cfg.optimizer
        )
        flux_estimator.to(device)

        # optimize
        trainer = pl.Trainer(**cfg.training.trainer)
        trainer.fit(flux_estimator, datamodule=dataset)

        # evaluate residuals
        # first define a data loader
        test_dataloader = dataset.test_dataloader()

        target = 0.0
        loss = 0.0
        counter = 0

        # not sure why I have to call this again
        flux_estimator.to(device)

        for _, batch in enumerate(test_dataloader):

            # loss at true parameters
            target += flux_estimator.kl_qp_flux_loss(
                batch,
                batch["fluxes"],
                # set the sd to 1
                batch["fluxes"] * 0.0 + 1.0,
            )[0].mean()

            # loss at estimated parameters
            out = flux_estimator(batch["images"])
            loss += flux_estimator.kl_qp_flux_loss(batch, out["mean"], out["sd"])[0].mean()

            counter += 1

        target /= counter
        loss /= counter

        print("Target loss: ", target)
        print("Estimate loss: ", loss)

        # check that estimated loss is within 10 %of true loss
        if devices.use_cuda:
            assert torch.abs(target - loss) / target < 0.1
