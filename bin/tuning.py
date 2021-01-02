import hydra
from omegaconf import DictConfig

import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from bliss import sleep
from bliss.datasets.simulated import SimulatedDataset


def SleepRayTune(search_space, cfg: DictConfig):
    # set up the config for SleepPhase
    cfg.model.encoder.params.enc_conv_c = search_space["enc_conv_c"]
    cfg.model.encoder.params.enc_kern = search_space["enc_kern"]
    cfg.model.encoder.params.enc_hidden = search_space["enc_hidden"]
    cfg.optimizer.params.lr = search_space["lr"]
    cfg.optimizer.params.weight_decay = search_space["weight_decay"]

    # model
    model = sleep.SleepPhase(cfg)

    # data module
    dataset = SimulatedDataset(cfg)

    # set up trainer
    trainer = pl.Trainer(
        weights_summary=None,
        max_epochs=cfg.tuning.max_epochs,
        gpus=cfg.tuning.gpus_per_trial,
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
# model=m2, dataset=m2, training=m2 optimizer=m2 in terminal
def main(cfg: DictConfig):

    logger = logging.getLogger()

    # restrict the number for cuda
    ray.init(num_gpus=cfg.tuning.allocated_gpus)

    search_space = {
        "enc_conv_c": tune.choice(list(cfg.tuning.search_space.enc_conv_c)),
        "enc_kern": tune.choice(list(cfg.tuning.search_space.enc_kern)),
        "enc_hidden": tune.choice(list(cfg.tuning.search_space.enc_hidden)),
        "lr": tune.loguniform(*cfg.tuning.search_space.lr),
        "weight_decay": tune.loguniform(*cfg.tuning.search_space.weight_decay),
    }

    # scheduler
    scheduler = ASHAScheduler(
        max_t=cfg.tuning.max_epochs,
        grace_period=cfg.tuning.grace_period,
    )
    # search algorithm
    search_alg = HyperOptSearch()
    search_alg = ConcurrencyLimiter(
        search_alg, max_concurrent=cfg.tuning.allocated_gpus
    )

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
        tune.with_parameters(SleepRayTune, cfg=cfg),
        resources_per_trial={"gpu": cfg.tuning.gpus_per_trial},
        num_samples=cfg.tuning.num_samples,
        verbose=cfg.tuning.verbose,
        config=search_space,
        scheduler=scheduler,
        metric="loss",
        mode="min",
        search_alg=search_alg,
        progress_reporter=reporter,
        name="tune_sleep",
    )

    best_config = trials.get_best_config(metric="loss", mode="min")
    logger.info(f"Best config: {best_config}")


if __name__ == "__main__":
    # sets seeds for numpy, torch, and python.random
    # TODO: Test reproducibility and decide wether to use `Trainer(deterministic=True)`, 10% slower
    pl.trainer.seed_everything(42)
    main()
