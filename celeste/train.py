from pathlib import Path
from abc import ABC, abstractmethod
import warnings
import time
import shutil

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl

from . import sleep


def set_seed(torch_seed, np_seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if torch_seed:
        torch.manual_seed(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if np_seed:
        np.random.seed(np_seed)


class TrainModel(ABC):
    def __init__(
        self,
        model,
        dataset,
        slen,
        n_bands,
        lr=1e-3,
        weight_decay=1e-5,
        batchsize=64,
        eval_every: int = None,
        out_dir=None,
        dloader_params=None,
        torch_seed=None,
        np_seed=None,
        verbose=False,
    ):
        set_seed(torch_seed, np_seed)  # seed for training.

        self.dataset = dataset
        self.slen = slen
        self.n_bands = n_bands
        self.batchsize = batchsize

        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        # evaluation and results
        self.verbose = verbose
        self.eval_every = eval_every

        self.out_dir = out_dir
        self.output_file, self.state_file_template = self.prepare_filepaths()

        self.model = model
        self.optimizer = self.get_optimizer()

        # data loader
        (
            self.train_loader,
            self.test_loader,
            self.size_train,
            self.size_test,
        ) = self.get_dloader(dloader_params)

    def get_dloader(self, dloader_params):
        if dloader_params is not None:
            assert "split" in dloader_params and "num_workers" in dloader_params

            split = dloader_params["split"]
            num_workers = dloader_params["num_workers"]
            assert (
                split > 0 or self.eval_every is None
            ), "Split is 0 then eval_every must be None."

            size_test = int(dloader_params["split"] * len(self.dataset))
            size_train = len(self.dataset) - size_test
            tt_split = int(split * len(self.dataset))
            test_indices = np.mgrid[:tt_split]  # 10% of data only is for test.
            train_indices = np.mgrid[tt_split : len(self.dataset)]

            train_loader = DataLoader(
                self.dataset,
                batch_size=self.batchsize,
                num_workers=num_workers,
                pin_memory=True,
                sampler=SubsetRandomSampler(train_indices),
            )

            test_loader = DataLoader(
                self.dataset,
                batch_size=self.batchsize,
                num_workers=num_workers,
                pin_memory=True,
                sampler=SubsetRandomSampler(test_indices),
            )

            return train_loader, test_loader, size_train, size_test

        else:
            return None, None, None, None

    def get_optimizer(self):
        optimizer = Adam(
            [{"params": self.model.parameters(), "lr": self.lr}],
            weight_decay=self.weight_decay,
        )
        return optimizer

    def prepare_filepaths(self):
        # need to provide out_dir directory for logging and state_dict saving to be available.
        if self.out_dir:
            if self.out_dir.exists():
                warnings.warn(
                    "The output directory already exists, previous results will be deleted"
                )
                shutil.rmtree(self.out_dir)

            self.out_dir.mkdir(exist_ok=False)

            state_file_template = self.out_dir.joinpath(
                "state_{}.dat"
            ).as_posix()  # insert epoch later.

            output_file = self.out_dir.joinpath(
                "output.txt"
            )  # save the output that is being printed.

            return output_file, state_file_template

        else:
            return None, None

    def write_to_output(self, text):
        if self.verbose:
            print(text)

        if self.output_file:
            with open(self.output_file, "a") as out:
                print(text, file=out)

    def run(self, n_epochs):

        for epoch in range(n_epochs):
            self.step(epoch, train=True)

            if (
                self.eval_every
                and (self.verbose or self.state_file_template)
                and epoch % self.eval_every == 0
            ):
                self.step(epoch, train=False)

    def step(self, epoch, train=True):
        # train or evaluate in the next epoch
        if train:
            self.model.train()
        else:
            self.model.eval()

        t0 = time.time()
        batch_generator = self.get_batch_generator()
        avg_loss = 0.0  # average results for this epoch
        avg_results = 0.0

        for batch in batch_generator:

            loss, results = self.get_loss_and_results(batch)
            avg_loss, avg_results = self.update_avg(
                avg_loss, avg_results, loss, results
            )

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if train and (self.output_file or self.verbose):
            self.log_train(epoch, avg_loss, avg_results, t0)
        elif not train:
            self.log_eval(epoch, avg_loss, avg_results)

    @abstractmethod
    def get_batch_generator(self):
        # return an iterator over all batches in a single epoch (if using data loader, then
        # this would be the dloader)
        pass

    @abstractmethod
    def get_loss_and_results(self, batch):
        # get all training/evaluation results from a batch, this includes loss and maybe other
        # useful things to log.
        pass

    @abstractmethod
    def update_avg(self, avg_loss, avg_results, loss, results):
        pass

    @abstractmethod
    def log_train(self, epoch, avg_loss, avg_results, t0):
        pass

    @abstractmethod
    def log_eval(self, epoch, avg_loss, avg_results):
        # usually save state dict here.
        pass


class SleepTraining(TrainModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = self.model

    @staticmethod
    def _get_params_from_batch(batch):
        return (
            batch["images"],
            batch["locs"],
            batch["galaxy_params"],
            batch["log_fluxes"],
            batch["galaxy_bool"],
        )

    def get_batch_generator(self):
        assert (
            len(self.dataset) % self.batchsize == 0
        ), "It's easier if they are multiples of each other"

        num_batches = int(len(self.dataset) / self.batchsize)
        for i in range(num_batches):
            yield self.dataset.get_batch(batchsize=self.batchsize)

    def get_loss_and_results(self, batch):
        # log_fluxes or gal_params are returned as true_source_params.
        # already in cuda.
        (
            images,
            true_locs,
            true_galaxy_params,
            true_log_fluxes,
            true_galaxy_bool,
        ) = self._get_params_from_batch(batch)

        # evaluate log q
        (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = sleep.get_inv_kl_loss(
            self.encoder,
            images,
            true_locs,
            true_galaxy_params,
            true_log_fluxes,
            true_galaxy_bool,
        )

        results = (
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        )

        return loss, results

    def update_avg(self, avg_loss, avg_results, loss, results):

        avg_loss += loss.item() * self.batchsize / len(self.dataset)

        tiles_per_epoch = len(self.dataset) * self.encoder.n_tiles
        avg_results += np.array([r.sum().item() / tiles_per_epoch for r in results])

        return avg_loss, avg_results

    def log_train(
        self, epoch, avg_loss, avg_results, t0,
    ):
        assert self.verbose or self.out_dir, "Not doing anything."

        # print and save train results.
        elapsed = time.time() - t0
        out_text = (
            f"{epoch} loss: {avg_loss:.4f}; counter loss: {avg_results[0]:.4f}; locs loss: "
            f"{avg_results[1]:.4f}; galaxy_params loss: {avg_results[2]:.4f}; "
            f"star_params_loss: {avg_results[3]:.4f}; galaxy_bool_loss: {avg_results[4]:.4f}"
            f"\t [{elapsed:.1f} seconds]"
        )

        self.write_to_output(out_text)

    def log_eval(self, epoch, avg_loss, avg_results):
        assert self.verbose or self.out_dir, "Not doing anything"

        out_text = (
            f"**** test loss: {avg_loss:.3f}; counter loss: {avg_results[0]:.3f}; "
            f"locs loss: {avg_results[1]:.3f}; galaxy param loss: {avg_results[2]:.3f} "
            f"star param loss: {avg_results[3]:.3f}; galaxy bool loss: {avg_results[4]:.3f}****"
        )

        self.write_to_output(out_text)

        if self.state_file_template and self.output_file:
            state_file = Path(self.state_file_template.format(epoch))
            state_text = (
                "**** writing the encoder parameters to "
                + state_file.as_posix()
                + " ***"
            )

            self.write_to_output(state_text)

            torch.save(self.encoder.state_dict(), state_file)


class SleepPhase(pl.LightningModule):
    def __init__(self, n_image, dataset, encoder, lr=1e-3, weight_decay=1e-5):
        """dataset is an SourceIterableDataset class"""
        super(SleepPhase, self).__init__()

        self.n_image = n_image
        self.dataset = dataset
        self.model = encoder
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self):
        pass

    def configure_optimizers(self):
        return Adam(
            [{"params": self.model.parameters(), "lr": self.lr}],
            weight_decay=self.weight_decay,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)

    def training_step(self, batch, batch_idx):
        (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = self.get_loss(batch)

        return {
            "loss": loss,
            "logs": {
                "training_loss": loss,
                "counter_loss": counter_loss,
                "locs_loss": locs_loss,
                "galaxy_params_loss": galaxy_params_loss,
                "star_params_loss": star_params_loss,
                "galaxy_bool_loss": galaxy_bool_loss,
            },
        }

    def training_epoch_end(self, outputs):
        avg_loss = 0
        avg_counter_loss = 0
        avg_locs_loss = 0
        avg_galaxy_params_loss = 0
        avg_star_params_loss = 0
        avg_galaxy_bool_loss = 0
        tiles_per_epoch = self.n_image * self.model.n_tiles

        for output in outputs:
            avg_loss += output["loss"] * len(outputs)
            avg_counter_loss += torch.sum(output["logs"]["counter_loss"]) * len(outputs)
            avg_locs_loss += torch.sum(output["logs"]["locs_loss"]) * len(outputs)
            avg_galaxy_params_loss += torch.sum(
                output["logs"]["galaxy_params_loss"]
            ) * len(outputs)
            avg_star_params_loss += torch.sum(output["logs"]["star_params_loss"]) * len(
                outputs
            )
            avg_galaxy_bool_loss += torch.sum(output["logs"]["galaxy_bool_loss"]) * len(
                outputs
            )

        avg_loss /= self.n_image
        avg_counter_loss /= tiles_per_epoch
        avg_locs_loss /= tiles_per_epoch
        avg_galaxy_params_loss /= tiles_per_epoch
        avg_star_params_loss /= tiles_per_epoch
        avg_galaxy_bool_loss /= tiles_per_epoch

        return {
            "log": {
                "train_loss": avg_loss,
                "counter_loss": avg_counter_loss,
                "locs_loss": avg_locs_loss,
                "galaxy_params_loss": avg_galaxy_params_loss,
                "star_params_loss": avg_star_params_loss,
                "galaxy_bool_loss": avg_galaxy_bool_loss,
            }
        }

    def get_loss(self, batch):
        (images, true_locs, true_galaxy_params, true_log_fluxes, true_galaxy_bool,) = (
            batch["images"],
            batch["locs"],
            batch["galaxy_params"],
            batch["log_fluxes"],
            batch["galaxy_bool"],
        )

        # evaluate log q
        (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = sleep.get_inv_kl_loss(
            self.model,
            images,
            true_locs,
            true_galaxy_params,
            true_log_fluxes,
            true_galaxy_bool,
        )

        return (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        )
