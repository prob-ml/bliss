import logging
import os

import hydra
import numpy as np
import pytorch_lightning as pl
import ray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch


def sleep_trainable(search_space, cfg: DictConfig):
    # set up the config for SleepPhase
    cfg.model.encoder.channel = search_space["channel"]
    cfg.model.encoder.hidden = search_space["hidden"]
    cfg.model.encoder.spatial_dropout = search_space["spatial_dropout"]
    cfg.model.encoder.dropout = search_space["dropout"]
    cfg.optimizer.kwargs.lr = search_space["lr"]
    cfg.optimizer.kwargs.weight_decay = search_space["weight_decay"]

    # model
    model = instantiate(cfg.model)

    # data module
    dataset = instantiate(cfg.dataset)

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
                    "loss": "val/loss",
                    "star_count_acc": "val/acc_counts",
                    "galaxy_counts_acc": "val/gal_counts",
                    "locs_mae": "val/locs_mae",
                    "fluxes_mae": "val/star_fluxes_mae",
                },
                on="validation_end",
            )
        ],
    )
    trainer.fit(model, datamodule=dataset)


# model=m2 dataset=m2 training=m2 optimizer=m2 in terminal
def tune(cfg: DictConfig, local_mode=False):
    # sets seeds for numpy, torch, and python.random
    # TODO: Test reproducibility and decide wether to use `Trainer(deterministic=True)`, 10% slower
    pl.trainer.seed_everything(cfg.tuning.seed)

    # restrict the number for cuda
    ray.init(num_gpus=cfg.tuning.allocated_gpus, local_mode=local_mode)

    discrete_search_space = {
        "channel": list(range(*cfg.tuning.search_space.channel)),
        "hidden": list(range(*cfg.tuning.search_space.hidden)),
    }

    search_space = {
        # Not as clean as tune.randint(*cfg.tuning...)
        # Work around solution so that these values are correctly displayed in tensorboard
        # This also creats primitive dtype supported by omegaconf
        "channel": ray.tune.choice(discrete_search_space["channel"]),
        "hidden": ray.tune.choice(discrete_search_space["hidden"]),
        "spatial_dropout": ray.tune.uniform(*cfg.tuning.search_space.spatial_dropout),
        "dropout": ray.tune.uniform(*cfg.tuning.search_space.dropout),
        "lr": ray.tune.loguniform(*cfg.tuning.search_space.lr),
        "weight_decay": ray.tune.loguniform(*cfg.tuning.search_space.weight_decay),
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

    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max(1, cfg.tuning.allocated_gpus))

    # define how to report the results
    reporter = CLIReporter(
        parameter_columns={
            "channel": "c",
            "hidden": "h",
            "dropout": "d",
            "spatial_dropout": "sd",
            "lr": "lr",
            "weight_decay": "wd",
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
    analysis = ray.tune.run(
        ray.tune.with_parameters(sleep_trainable, cfg=cfg),
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
        OmegaConf.save(conf, hydra.utils.to_absolute_path(cfg.tuning.best_config_save_path))
