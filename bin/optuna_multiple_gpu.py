import os
import optuna
import hydra
from omegaconf import DictConfig

from bliss.hyperparameter import SleepTune


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    # where to save the checkpoint
    model_dir = os.getcwd()

    # which metric to monitor
    monitor = "val_loss"

    # set up pruner
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

    # how many gpus do we need
    n_gpu = 3

    # set up the config for the training
    cfg.model.encoder.params.update(
        {
            "enc_conv_c": (5, 25, 5),
            "enc_hidden": (64, 128, 64),
        }
    )

    cfg.optimizer.params.update({"lr": (1e-4, 1e-2), "weight_decay": (1e-6, 1e-4)})

    SleepTune(
        cfg,
        max_epochs=100,
        model_dir=model_dir,
        monitor=monitor,
        n_batches=4,
        batch_size=32,
        direction="minimize",
        data_seed=10,
        n_trials=100,
        time_out=600,
        gc_after_trial=True,
        sampler=None,  # use the default sampler of the optuna
        pruner=pruner,
        num_gpu=n_gpu,
        storage="sqlite:///zz.db",
    )

    print("Tunning is finished")


if __name__ == "__main__":
    main()
