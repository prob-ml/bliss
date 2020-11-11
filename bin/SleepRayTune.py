import shutil
import hydra
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from bliss import sleep
from bliss.datasets.simulated import SimulatedModule


# define the callback that monitors val_loss
callback = TuneReportCallback({"loss": "val_loss"}, on="validation_end")


def SleepRayTune(config, cfg: DictConfig, num_epochs, num_gpu):
    # set up the config for SleepPhase
    cfg.model.encoder.params["enc_conv_c"] = config["enc_conv_c"]
    cfg.model.encoder.params["enc_kern"] = config["enc_kern"]
    cfg.model.encoder.params["enc_hidden"] = config["enc_hidden"]
    cfg.optimizer.params["lr"] = config["lr"]
    cfg.optimizer.params["weight_decay"] = config["weight_decay"]

    # model
    model = sleep.SleepPhase(cfg)

    # data module
    dataset = SimulatedModule(cfg=cfg)

    # parameters for trainer
    num_epochs = num_epochs
    num_gpu = num_gpu

    torch.cuda.set_device("cuda")

    # set up trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpu,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {"loss": "val_loss"},
                on="validation_end",
            )
        ],
    )
    trainer.fit(model, datamodule=dataset)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig, num_epochs=100, gpus_per_trial=1):
    config = {
        "enc_conv_c": tune.choice([5, 15, 20]),
        "enc_kern": tune.choice([3, 5, 7]),
        "enc_hidden": tune.choice([64, 128]),
        "lr": tune.choice([1e-4, 1e-2]),
        "weight_decay": tune.choice([1e-6, 1e-4]),
    }

    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=num_epochs, grace_period=1
    )

    reporter = CLIReporter(
        parameter_columns=[
            "enc_conv_c",
            "enc_kern",
            "enc_hidden",
            "lr",
            "weight_decay",
        ],
        metric_columns=["loss"],
    )

    tune.run(
        tune.with_parameters(
            SleepRayTune,
            cfg=cfg,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
        ),
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="SleepRayTune",
    )

    # shutil.rmtree(data_dir)


if __name__ == "__main__":
    main()
