import os
import numpy as np
import logging

import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

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
    logging.getLogger("lightning").setLevel(0)
    trainer = pl.Trainer(
        limit_val_batches=cfg.tuning.limit_val_batches,
        weights_summary=None,
        max_epochs=cfg.tuning.n_epochs,
        gpus=cfg.tuning.gpus_per_trial,
        logger=False,
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "val_detection_loss",
                    "star_count_acc": "val_acc_counts",
                    "galaxy_counts_acc": "val_gal_counts",
                    "locs_mae": "val_locs_mae",
                    "fluxes_mae": "val_star_fluxes_mae",
                },
                on="validation_end",
            )
        ],
    )
    trainer.fit(model, datamodule=dataset)


# model=m2 dataset=m2 training=m2 optimizer=m2 in terminal
def main(cfg: DictConfig, local_mode=False):
    # sets seeds for numpy, torch, and python.random
    # TODO: Test reproducibility and decide wether to use `Trainer(deterministic=True)`, 10% slower
    pl.trainer.seed_everything(cfg.tuning.seed)

    # restrict the number for cuda
    ray.init(num_gpus=cfg.tuning.allocated_gpus, local_mode=local_mode)

    discrete_search_space = {
        "enc_conv_c": list(range(*cfg.tuning.search_space.enc_conv_c)),
        "enc_kern": list(range(*cfg.tuning.search_space.enc_kern)),
        "enc_hidden": list(range(*cfg.tuning.search_space.enc_hidden)),
    }

    search_space = {
        # Not as clean as tune.randint(*cfg.tuning...)
        # Work around solution so that these values are correctly displayed in tensorboard
        # This also creats primitive dtype supported by omegaconf
        "enc_conv_c": tune.choice(discrete_search_space["enc_conv_c"]),
        "enc_kern": tune.choice(discrete_search_space["enc_kern"]),
        "enc_hidden": tune.choice(discrete_search_space["enc_hidden"]),
        "lr": tune.loguniform(*cfg.tuning.search_space.lr),
        "weight_decay": tune.loguniform(*cfg.tuning.search_space.weight_decay),
    }

    # scheduler
    scheduler = ASHAScheduler(
        max_t=cfg.tuning.n_epochs,
        grace_period=cfg.tuning.grace_period,
    )

    # search algorithm
    # set last best result as intial value if exists
    if os.path.exists(hydra.utils.to_absolute_path(cfg.tuning.best_config_save_path)):
        last_best_result = OmegaConf.load(
            hydra.utils.to_absolute_path(cfg.tuning.best_config_save_path)
        )
        last_best_config = OmegaConf.to_container(last_best_result.config)

        # change to index-based value as required by hyperopt
        for k, v in discrete_search_space.items():
            index = np.flatnonzero(np.array(v) == last_best_config[k])
            last_best_config[k] = index.item()

        print("\nSet intial starting point for search algorithm as:")
        print(OmegaConf.to_yaml(last_best_result.config), "\n")

        search_alg = HyperOptSearch(
            points_to_evaluate=[last_best_config], random_state_seed=cfg.tuning.seed
        )
    else:
        search_alg = HyperOptSearch(random_state_seed=cfg.tuning.seed)

    search_alg = ConcurrencyLimiter(
        search_alg, max_concurrent=cfg.tuning.max_concurrent
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
    # TODO add stop criterion for nan loss
    analysis = tune.run(
        tune.with_parameters(sleep_trainable, cfg=cfg),
        resources_per_trial={"gpu": cfg.tuning.gpus_per_trial},
        num_samples=cfg.tuning.n_samples,
        verbose=cfg.tuning.verbose,
        config=search_space,
        scheduler=scheduler,
        metric="loss",
        mode="min",
        local_dir=hydra.utils.to_absolute_path(cfg.tuning.log_path),
        search_alg=search_alg,
        progress_reporter=reporter,
        name="tune_sleep",
    )

    if cfg.tuning.save:
        best_result = analysis.best_result
        conf = OmegaConf.create(best_result)
        OmegaConf.save(
            conf, hydra.utils.to_absolute_path(cfg.tuning.best_config_save_path)
        )


if __name__ == "__main__":
    main()
