import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
import os

import pytorch_lightning as pl

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from bliss import sleep
from bliss.datasets.simulated import SimulatedDataset


def sleep_trainable(search_space, cfg: DictConfig):
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
        max_epochs=cfg.tuning.n_epochs,
        gpus=cfg.tuning.gpus_per_trial,
        logger=False,
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
    # sets seeds for numpy, torch, and python.random
    # TODO: Test reproducibility and decide wether to use `Trainer(deterministic=True)`, 10% slower
    pl.trainer.seed_everything(cfg.tuning.seed)

    assert hydra.utils.get_original_cwd().endswith(
        "/bin"
    ), f"This script needs to be run in /bin instead of {hydra.utils.get_original_cwd()}"
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
        max_t=cfg.tuning.n_epochs,
        grace_period=cfg.tuning.grace_period,
    )
    # search algorithm
    search_alg = HyperOptSearch()
    search_alg = ConcurrencyLimiter(
        search_alg, max_concurrent=cfg.tuning.allocated_gpus
    )

    # define how to report the results
    reporter = CLIReporter(
        parameter_columns={
            "enc_conv_c": "conv_c",
            "enc_kern": "kern",
            "enc_hidden": "hidden",
            "lr": "lr",
            "weight_decay": "weight_decay",
        },
        metric_columns={
            "loss": "loss",
            "star_count_accuracy": "star_ct_acc",
            "galaxy_counts_acc": "gal_ct_acc",
            "locs_median_mse": "loc_med_mse",
            "fluxes_avg_err": "flux_avg_err",
        },
    )

    # run the trials
    analysis = tune.run(
        tune.with_parameters(sleep_trainable, cfg=cfg),
        resources_per_trial={"gpu": cfg.tuning.gpus_per_trial},
        num_samples=cfg.tuning.n_samples,
        verbose=cfg.tuning.verbose,
        config=search_space,
        scheduler=scheduler,
        metric="loss",
        mode="min",
        local_dir=hydra.utils.to_absolute_path("../outputs/tuning"),
        search_alg=search_alg,
        progress_reporter=reporter,
        name="tune_sleep",
    )

    conf = OmegaConf.create(analysis.best_result)
    OmegaConf.save(conf, hydra.utils.to_absolute_path(cfg.tuning.best_config_save_path))


if __name__ == "__main__":
    main()
