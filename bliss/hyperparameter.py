import os
import re
import shutil
from typing import Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
import GPUtil

import multiprocessing as mp
from joblib import parallel_backend

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from omegaconf import DictConfig

import logging

from bliss.sleep import SleepPhase
from bliss.datasets.simulated import SimulatedModule


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
        data_seed: int = 10,
        single_gpu=None,
        gpu_queue: Optional = None,
    ):

        self.cfg = cfg

        # parameters for dataset simulation
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

        # set up for gpu
        self.single_gpu = single_gpu
        self.gpu_queue = gpu_queue

    def __call__(self, trial):

        # set seed
        torch.manual_seed(self.seed)

        # set up environment device
        if self.gpu_queue is not None:
            gpu_id = self.gpu_queue.get()
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
        elif self.gpu_queue is None and self.single_gpu is not None:
            device = torch.device(f"cuda:{self.single_gpu}")
            torch.cuda.set_device(device)

        star_dataset = SimulatedModule(cfg=self.cfg)

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

        lr = (
            trial.suggest_loguniform("learning rate", self.lr[0], self.lr[1])
            if type(self.lr) is not float
            else self.lr
        )
        weight_decay = (
            trial.suggest_loguniform(
                "weight_decay", self.weight_decay[0], self.weight_decay[1]
            )
            if type(self.weight_decay) is not float
            else self.weight_decay
        )

        # set up the optimizer base on learning rate and weight decay
        self.cfg.optimizer.params.update(dict(lr=lr, weight_decay=weight_decay))

        # set up the checkpoint callback for the model
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(self.model_dir, "trial_{}".format(trial.number), "{epoch}"),
            monitor=self.monitor,
        )

        # Initiate the model
        model = SleepPhase(self.cfg)

        # put correct device to model
        use_gpu = 0
        if self.gpu_queue is not None and self.single_gpu is None:
            use_gpu = [gpu_id]
        elif self.gpu_queue is None and self.single_gpu is not None:
            use_gpu = [self.single_gpu]

        # put gpu id back to the queue
        if self.gpu_queue is not None:
            self.gpu_queue.put(gpu_id)

        # set up the trainer
        # PyTorchLightningPruningCallback allow the pruner to stop early
        trainer = pl.Trainer(
            logger=False,
            gpus=use_gpu,
            checkpoint_callback=checkpoint_callback,
            max_epochs=self.max_epochs,
            callbacks=[
                self.metrics_callback,
                PyTorchLightningPruningCallback(trial, monitor=self.monitor),
            ],
        )

        # start training
        trainer.fit(model, datamodule=star_dataset)

        return self.metrics_callback.metrics[-1][self.monitor].item()


def SleepTune(
    cfg: DictConfig,
    max_epochs: int,
    model_dir,
    monitor,
    direction: str = "minimize",
    data_seed: int = 10,
    n_trials=100,
    time_out=600,
    gc_after_trial: bool = False,
    sampler=None,
    pruner=None,
    num_gpu: int or list = 2,
    storage=None,
):
    """
    :param cfg: DictConfig from hydra
    :param max_epochs: in integer to indicate epochs of training
    :param model_dir: where to save the model checkpoint of lightning
    :param monitor: which metric to monitor for the hyperparameter selection
    :param n_batches: how many batches per epoch
    :param batch_size: the size of image simulated for each batch
    :param direction: minimize or maximize the chosen metric
    :param data_seed: set a seed for data simulation
    :param n_trials: the maximum trials to run
    :param time_out: the maximum time to run for each trial
    :param gc_after_trial: enable garbage collection after each trial?
    :param sampler: optuna.sampler
    :param pruner: optuna.pruner
    :param num_gpu: numer of GPUs to use. Could be an integer or a list of ids.
                    CPU only: set this arugument to be zero
                    Single GPU: either pass in 1 ot a list single id, e.g. [1]
                    Multiple GPUs: either pass in an integer > 1 and SleepTune pick
                                    available GPUS, or a list of ids.
    :param storage: If using multiple GPUs, storage must be defined as optuna.RDBStorage

    :return: an optuna.trial.FrozenTrial object
    """

    # set up number of gpu to use
    single_gpu = None
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

    # if using multiple GPU, storage must be define by optuna.RDBStorage
    if gpu_queue is not None and storage is None:
        raise AttributeError("storage must defined when using multiple gpu")

    # set up object class
    object = SleepObjective(
        cfg,
        max_epochs,
        model_dir,
        MetricsCallback(),
        monitor,
        data_seed,
        single_gpu=single_gpu,
        gpu_queue=gpu_queue,
    )

    if object.gpu_queue is None:
        logger.info("Starting single GPU training")

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            object,
            n_trials=n_trials,
            timeout=time_out,
            gc_after_trial=gc_after_trial,
        )

    else:
        logger.info("Starting multiple GPU training. One GPU per trial")

        study = optuna.create_study(
            direction=direction,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )

        n_jobs = len(num_gpu) if type(num_gpu) is list else num_gpu
        with parallel_backend("loky", n_jobs=n_jobs):
            logger.info(f"Use loky backend and spawn {n_jobs} workers")
            study.optimize(
                object,
                n_jobs=n_jobs,
                n_trials=n_trials,
                timeout=time_out,
                gc_after_trial=gc_after_trial,
            )

    # get the best trial
    trial = study.best_trial

    logger.info("Number of finished trials: {}".format(len(study.trials)))
    logger.info(f"Best trial: {trial.number} with value {trial.value}")
    print("  The params for the best trial: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # remove the trial folders created by optuna
    trial_re = re.compile(r"trial_\d*")
    for root, dirs, files in os.walk(model_dir):
        for dir in dirs:
            if trial_re.match(dir):
                shutil.rmtree(os.path.join(root, dir))

    return trial
