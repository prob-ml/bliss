import hydra
from omegaconf import DictConfig

import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ray import tune
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from bliss import sleep
from bliss.datasets.simulated import SimulatedDataset


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
    dataset = SimulatedDataset(cfg=cfg)

    # parameters for trainer
    num_epochs = num_epochs
    num_gpu = num_gpu

    # set up trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpu,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "val_loss",
                    "star_count_acc": "val_acc_counts",
                    "galaxy_counts_acc": "val_gal_counts",
                    "locs_median_mse": "val_locs_median_mse",
                    "fluxes_avg_err": "val_fluxes_avg_err",
                },
                on="validation_end",
            )
        ],
    )
    trainer.fit(model, datamodule=dataset)


@hydra.main(config_path="../config", config_name="config")
# model=tuning, dataset=m2, training=tuning optimizer=m2 in terminal
def main(cfg: DictConfig, num_epochs=200, gpus_per_trial=1):

    logger = logging.getLogger()

    # restrict the number for cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training.multigpus

    # define the parameter space
    config = {
        "enc_conv_c": tune.grid_search([10, 15, 20, 25, 30]),
        "enc_kern": tune.grid_search([3]),
        "enc_hidden": tune.grid_search([64, 128, 192, 256, 320]),
        "lr": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
    }

    # scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=num_epochs,
        grace_period=cfg.training.grace_period,
    )

    # search algorithm
    # df_search = DragonflySearch(
    #    optimizer="bandit", domain="euclidean", metric="loss", mode="min"
    # )

    # define how to report the results
    reporter = CLIReporter(
        parameter_columns=[
            "enc_conv_c",
            "enc_kern",
            "enc_hidden",
            "lr",
            "weight_decay",
        ],
        metric_columns=[
            "loss",
            "star_count_accuracy",
            "galaxy_counts_acc",
            "locs_median_mse",
            "fluxes_avg_err",
        ],
    )

    # run the trials
    trials = tune.run(
        tune.with_parameters(
            SleepRayTune,
            cfg=cfg,
            num_epochs=num_epochs,
            num_gpu=gpus_per_trial,
        ),
        resources_per_trial={"gpu": gpus_per_trial},
        num_samples=20,
        verbose=1,
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_sleep",
    )

    best_config = trials.get_best_config(metric="loss", mode="min")
    logger.info(f"Best config: {best_config}")


if __name__ == "__main__":
    # sets seeds for numpy, torch, and python.random
    pl.trainer.seed_everything(42)
    main()
