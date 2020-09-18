import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from multiprocessing import Manager
from joblib import parallel_backend

from bliss.sleep import SleepPhase


class SleepObjective(object):
    def __init__(
        self,
        dataset,
        encoder_kwargs: dict,
        max_epochs: int,
        lr: tuple,
        weight_decay: tuple,
        model_dir,
        metrics_callback,
        monitor,
        single_gpu_id,
        gpu_queue=None,
    ):
        self.dataset = dataset

        assert type(encoder_kwargs["enc_conv_c"]) is tuple
        assert type(encoder_kwargs["enc_hidden"]) is tuple
        assert (
            len(encoder_kwargs["enc_conv_c"]) == 3
            and len(encoder_kwargs["enc_hidden"]) == 3
        )
        self.encoder_kwargs = encoder_kwargs
        self.enc_conv_c_min = self.encoder_kwargs["enc_conv_c"][0]
        self.enc_conv_c_max = self.encoder_kwargs["enc_conv_c"][1]
        self.enc_conv_c_int = self.encoder_kwargs["enc_conv_c"][2]

        self.enc_hidden_min = self.encoder_kwargs["enc_hidden"][0]
        self.enc_hidden_max = self.encoder_kwargs["enc_hidden"][1]
        self.enc_hidden_int = self.encoder_kwargs["enc_hidden"][2]

        assert type(lr) is tuple
        assert type(weight_decay) is tuple
        assert len(lr) == 2 and len(weight_decay) == 2
        self.lr = lr
        self.weight_decay = weight_decay

        self.max_epochs = max_epochs
        self.model_dir = model_dir
        self.metrics_callback = metrics_callback
        self.monitor = monitor

        # set up for multiple gpu
        self.gpu_queue = gpu_queue

        # set up for single gpu
        self.single_gpu_id = single_gpu_id

    def __call__(self, trial):
        if self.gpu_queue is not None:
            gpu_id = self.gpu_queue.get()

        self.encoder_kwargs["enc_conv_c"] = trial.suggest_int(
            "enc_conv_c",
            self.enc_conv_c_min,
            self.enc_conv_c_max,
            self.enc_conv_c_int,
        )

        self.encoder_kwargs["enc_hidden"] = trial.suggest_int(
            "enc_hidden",
            self.enc_hidden_min,
            self.enc_hidden_max,
            self.enc_hidden_int,
        )

        lr = trial.suggest_loguniform("learning rate", self.lr[0], self.lr[1])
        weight_decay = trial.suggest_loguniform(
            "weight_decay", self.weight_decay[0], self.weight_decay[1]
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            self.model_dir.joinpath("trial_{}".format(trial.number), "{epoch}"),
            monitor="val_loss",
        )

        model = SleepPhase(self.dataset, self.encoder_kwargs, lr, weight_decay)

        trainer = pl.Trainer(
            logger=False,
            gpus=[gpu_id] if self.gpu_queue is not None else self.single_gpu_id,
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


def multi_gpu_optuna(gpu_num, storage: str, direction: str, *sleepobjectiveargs):
    gpu_queue = Manager().Queue()

    study = optuna.create_study(storage=storage, direction=direction)
    # set up queue of devices
    for i in range(gpu_num):
        gpu_queue.put(i)

    sleepobjectiveargs.gpu_queue = gpu_queue

    with parallel_backend("multiprocessing", n_jobs=gpu_num):
        study.optimize(SleepObjective(*sleepobjectiveargs), n_trials=10, n_jobs=gpu_num)
