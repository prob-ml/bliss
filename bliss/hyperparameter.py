import os
from typing import Optional
from typing import Union
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
import GPUtil

import multiprocessing as mp
from multiprocessing.managers import SyncManager
from joblib import parallel_backend

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna import storages
from optuna import pruners
from optuna import samplers

from omegaconf import DictConfig

import logging

from bliss.sleep import SleepPhase
from bliss.datasets.simulated import SimulatedDataset


logger = logging.getLogger()

# define a callback to get matric from validation_end step
#  idea is same for the early stopping
class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


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
        data_seed: int = 10,
        single_gpu_id=1,
        gpu_queue: Optional[SyncManager] = None,
    ):

        self.cfg = cfg

        # parameters for dataset simulation
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.seed = data_seed  # user can pick a seed for data simulation

        # set up the encoder parameter search range
        self.enc_conv_c = cfg.model.encoder.params.enc_conv_c
        self.enc_kern = cfg.model.encoder.params.enc_kern
        self.enc_hidden = cfg.model.encoder.params.enc_hidden

        # set up the learning rate and weight decay
        self.lr = cfg.optimizer.params.lr
        self.weight_decay = cfg.optimizer.params.weight_decay

        # parameters for model training
        self.max_epochs = max_epochs
        self.model_dir = model_dir
        self.metrics_callback = metrics_callback
        self.monitor = monitor

        # set up for single gpu
        self.single_gpu = single_gpu_id

        # set up for multiple gpu
        self.gpu_queue = gpu_queue

    def __call__(self, trial):

        # set seed
        torch.manual_seed(self.seed)

        # set up environment device
        if self.gpu_queue is not None:
            gpu_id = self.gpu_queue.get()
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
        elif self.gpu_queue is None and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.single_gpu}")
            torch.cuda.set_device(device)

        # set up the dataset simulation
        self.cfg.dataset.params.update(
            {"n_batches": self.n_batches, "batch_size": self.batch_size}
        )

        star_dataset = SimulatedDataset(cfg=self.cfg)

        # update hydra config to be the search space
        self.cfg.model.encoder.params["enc_conv_c"] = (
            trial.suggest_int(
                "enc_conv_c",
                self.enc_conv_c[0],
                self.enc_conv_c[1],
                self.enc_conv_c[2],
            )
            if type(self.enc_conv_c) is not int
            else self.enc_conv_c
        )

        self.cfg.model.encoder.params["enc_kern"] = (
            trial.suggest_int(
                "enc_kern",
                self.enc_kern[0],
                self.enc_kern[1],
                self.enc_kern[2],
            )
            if type(self.enc_kern) is not int
            else self.enc_kern
        )

        self.cfg.model.encoder.params["enc_hidden"] = (
            trial.suggest_int(
                "enc_hidden",
                self.enc_hidden[0],
                self.enc_hidden[1],
                self.enc_hidden[2],
            )
            if type(self.enc_hidden) is not int
            else self.enc_hidden
        )

        lr = trial.suggest_loguniform("learning rate", self.lr[0], self.lr[1])
        weight_decay = trial.suggest_loguniform(
            "weight_decay", self.weight_decay[0], self.weight_decay[1]
        )

        # set up the optimizer base on learning rate and weight decay
        self.cfg.optimizer.params.update(dict(lr=lr, weight_decay=weight_decay))

        # set up the checkpoint callback for the model
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(self.model_dir, "trial_{}".format(trial.number), "{epoch}"),
            monitor=self.monitor,
        )

        # Initiate the model
        model = SleepPhase(self.cfg, star_dataset)

        # put correct device to model
        use_gpu = [gpu_id] if self.gpu_queue is not None else [self.single_gpu]
        use_cpu = 0

        # put gpu id back to the queue
        if self.gpu_queue is not None:
            self.gpu_queue.put(gpu_id)

        # set up the trainer
        ## PyTorchLightningPruningCallback allow the pruner to stop early
        trainer = pl.Trainer(
            logger=False,
            gpus=use_gpu if torch.cuda.is_available() else use_cpu,
            checkpoint_callback=checkpoint_callback,
            max_epochs=self.max_epochs,
            callbacks=[
                self.metrics_callback,
                PyTorchLightningPruningCallback(trial, monitor=self.monitor),
            ],
        )

        # start training
        trainer.fit(model)

        return self.metrics_callback.metrics[-1][self.monitor].item()


class SleepTune(object):
    def __init__(
        self,
        cfg: DictConfig,
        max_epochs: int,
        model_dir,
        monitor,
        n_batches,
        batch_size,
        direction: str = "minimum",
        data_seed: int = 10,
        n_trials=100,
        time_out=600,
        gc_after_trial: bool = False,
        sampler: Optional["samplers.BaseSampler"] = None,
        pruner: Optional[pruners] = None,
        num_gpu: int or list = 2,
        storage: Optional[Union[str, storages.BaseStorage]] = None,
    ):

        # set up number of gpu to use
        single_gpu = 1
        gpu_queue = None

        if num_gpu == 1:
            deviceID = GPUtil.getAvailable()
            single_gpu = deviceID[0]
        elif isinstance(num_gpu, list) and len(num_gpu) == 1:
            single_gpu = num_gpu[0]
        elif num_gpu > 1:
            deviceID = GPUtil.getAvailable(limit=num_gpu)
            # check if available GPUs are enough
            if len(deviceID) < num_gpu:
                raise Warning("Available GPU is less than gpu number specified")
            gpu_queue = mp.Manager().Queue()
            for i in deviceID:
                gpu_queue.put(i)
        elif isinstance(num_gpu, list) and len(num_gpu) > 1:
            gpu_queue = mp.Manager().Queue()
            for i in num_gpu:
                gpu_queue.put(i)

        # set up object class
        self.object = SleepObjective(
            cfg,
            max_epochs,
            model_dir,
            MetricsCallback(),
            monitor,
            n_batches,
            batch_size,
            data_seed,
            single_gpu_id=single_gpu,
            gpu_queue=gpu_queue,
        )

        # parameters for study object
        self.sampler = sampler
        self.pruner = pruner
        self.direction = direction

        if gpu_queue is not None and storage is None:
            raise AttributeError("storage must defined when using multiple gpu")
        self.storage = storage

        # parameters for study.optimize
        self.n_trials = n_trials
        self.time_out = time_out
        self.gc_after_trial = gc_after_trial

    def run(self):
        if self.object.gpu_queue is None:
            logger.info("Starting single GPU training")

            study = optuna.create_study(
                direction=self.direction,
                sampler=self.sampler,
                pruner=self.pruner,
            )

            study.optimize(
                self.object,
                n_trials=self.n_trials,
                timeout=self.time_out,
                gc_after_trial=self.gc_after_trial,
            )

        else:
            logger.info("Starting multiple GPU training. One GPU per trial")

            study = optuna.create_study(
                direction=self.direction,
                storage=self.storage,
                sampler=self.sampler,
                pruner=self.pruner,
            )

            with parallel_backend("loky", n_jobs=len(self.object.gpu_queue)):
                logger.info(
                    f"Use loky backend and spawn {len(self.object.gpu_queue)} workers"
                )
                study.optimize(
                    self.object,
                    n_jobs=len(self.object.gpu_queue),
                    n_trials=self.n_trials,
                    timeout=self.time_out,
                    gc_after_trial=self.gc_after_trial,
                )

        trial = study.best_trial

        logger.info("Number of finished trials: {}".format(len(study.trials)))
        logger.info(f"Best trial: {trial.number} with value {trial.value}")
        print("  The params for the best trial: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return trial
