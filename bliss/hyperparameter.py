import os
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import warnings
from optuna.integration import PyTorchLightningPruningCallback

from bliss.sleep import SleepPhase
from bliss.datasets.simulated import SimulatedDataset


class SleepObjective(object):
    def __init__(
        self,
        cfg: DictConfig,
        max_epochs: int,
        model_dir,
        metrics_callback,
        monitor,
        n_batches,
        batch_size,
        single_gpu_id=1,
        gpu_queue=None,
    ):

        self.cfg = cfg
        self.n_batches = n_batches
        self.batch_size = batch_size

        self.enc_conv_c_min = cfg.model.encoder.params.enc_conv_c[0]
        self.enc_conv_c_max = cfg.model.encoder.params.enc_conv_c[1]
        self.enc_conv_c_int = cfg.model.encoder.params.enc_conv_c[2]

        self.enc_hidden_min = cfg.model.encoder.params.enc_hidden[0]
        self.enc_hidden_max = cfg.model.encoder.params.enc_hidden[1]
        self.enc_hidden_int = cfg.model.encoder.params.enc_hidden[2]

        self.lr = cfg.optimizer.params.lr
        self.weight_decay = cfg.optimizer.params.weight_decay

        self.max_epochs = max_epochs
        self.model_dir = model_dir
        self.metrics_callback = metrics_callback
        self.monitor = monitor

        # set up for single gpu
        self.single_gpu = single_gpu_id

        # set up for multiple gpu
        self.gpu_queue = gpu_queue

    def __call__(self, trial):

        torch.manual_seed(10)

        if self.gpu_queue is not None:
            gpu_id = self.gpu_queue.get()
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
        elif self.gpu_queue is None and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.single_gpu}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        self.cfg.dataset.params.update(
            {"n_batches": self.n_batches, "batch_size": self.batch_size}
        )

        star_dataset = SimulatedDataset(cfg=self.cfg)

        self.cfg.model.encoder.params["enc_conv_c"] = trial.suggest_int(
            "enc_conv_c",
            self.enc_conv_c_min,
            self.enc_conv_c_max,
            self.enc_conv_c_int,
        )

        self.cfg.model.encoder.params["enc_hidden"] = trial.suggest_int(
            "enc_hidden",
            self.enc_hidden_min,
            self.enc_hidden_max,
            self.enc_hidden_int,
        )

        lr = trial.suggest_loguniform("learning rate", self.lr[0], self.lr[1])
        weight_decay = trial.suggest_loguniform(
            "weight_decay", self.weight_decay[0], self.weight_decay[1]
        )
        self.cfg.optimizer.params.update(dict(lr=lr, weight_decay=weight_decay))

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(self.model_dir, "trial_{}".format(trial.number), "{epoch}"),
            monitor="val_loss",
        )

        model = SleepPhase(self.cfg, star_dataset)

        # put correct device to model
        use_gpu = [gpu_id] if self.gpu_queue is not None else [self.single_gpu]
        use_cpu = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # code that produces a warning
            trainer = pl.Trainer(
                logger=False,
                gpus=use_gpu if torch.cuda.is_available() else use_cpu,
                checkpoint_callback=checkpoint_callback,
                max_epochs=self.max_epochs,
                callbacks=[self.metrics_callback],
                early_stop_callback=PyTorchLightningPruningCallback(
                    trial, monitor=self.monitor
                ),
            )

            trainer.fit(model)

        if self.gpu_queue is not None:
            self.gpu_queue.put(gpu_id)

        return self.metrics_callback.metrics[-1][self.monitor].item()
