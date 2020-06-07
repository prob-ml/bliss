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
        avg_results = None  # average results for this epoch

        for batch in batch_generator:

            results = self.get_results(batch)
            loss = self.get_loss(results)
            avg_results = self.update_avg(avg_results, results)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if train and (self.output_file or self.verbose):
            self.log_train(epoch, avg_results, t0)
        elif not train:
            self.log_eval(epoch, avg_results)

    @abstractmethod
    def get_batch_generator(self):
        # return an iterator over all batches in a single epoch (if using data loader, then
        # this would be the dloader)
        pass

    @abstractmethod
    def get_results(self, batch):
        # get all training/evaluation results from a batch, this includes loss and maybe other
        # useful things to log.
        pass

    @abstractmethod
    def update_avg(self, avg_results, results):
        pass

    @abstractmethod
    def get_loss(self, results):
        # get loss from results to pass propagate, etc.
        pass

    @abstractmethod
    def log_train(self, epoch, avg_results, t0):
        pass

    @abstractmethod
    def log_eval(self, epoch, avg_results):
        # usually save state dict here.
        pass


class SleepTraining(TrainModel):
    def __init__(self, *args, n_source_params, lr=1e-3, weight_decay=1e-5, **kwargs):
        super().__init__(*args, lr=lr, weight_decay=weight_decay, **kwargs)
        self.encoder = self.model
        self.n_source_params = n_source_params
        assert self.n_source_params == self.encoder.n_source_params

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

    def get_loss(self, results):
        return results[0]

    def get_results(self, batch):
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
        loss, counter_loss, locs_loss, source_params_loss = sleep.get_inv_kl_loss(
            self.encoder,
            images,
            true_locs,
            true_galaxy_params,
            true_log_fluxes,
            true_galaxy_bool,
        )

        return loss, counter_loss, locs_loss, source_params_loss

    def update_avg(self, avg_results, results):
        if avg_results is None:
            avg_loss, avg_counter_loss, avg_locs_loss, avg_source_params_loss = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
        else:
            (
                avg_loss,
                avg_counter_loss,
                avg_locs_loss,
                avg_source_params_loss,
            ) = avg_results

        loss, counter_loss, locs_loss, source_params_loss = results

        avg_loss += loss.item() * self.batchsize / len(self.dataset)

        tiles_per_epoch = len(self.dataset) * self.encoder.n_tiles
        avg_counter_loss += counter_loss.sum().item() / tiles_per_epoch
        avg_source_params_loss += source_params_loss.sum().item() / tiles_per_epoch
        avg_locs_loss += locs_loss.sum().item() / tiles_per_epoch

        return avg_loss, avg_counter_loss, avg_locs_loss, avg_source_params_loss

    def log_train(
        self, epoch, avg_results, t0,
    ):
        assert self.verbose or self.out_dir, "Not doing anything."

        avg_loss, counter_loss, locs_loss, source_param_loss = avg_results
        # print and save train results.
        elapsed = time.time() - t0
        out_text = (
            f"{epoch} loss: {avg_loss:.4f}; counter loss: {counter_loss:.4f}; locs loss: "
            f"{locs_loss:.4f}; source_params loss: {source_param_loss:.4f} \t [{elapsed:.1f} "
            f"seconds]"
        )

        self.write_to_output(out_text)

    def log_eval(self, epoch, avg_results):
        assert self.verbose or self.out_dir, "Not doing anything"

        (
            test_loss,
            test_counter_loss,
            test_locs_loss,
            test_source_param_loss,
        ) = avg_results
        out_text = (
            f"**** test loss: {test_loss:.3f}; counter loss: {test_counter_loss:.3f}; "
            f"locs loss: {test_locs_loss:.3f}; source param loss: {test_source_param_loss:.3f} ****"
        )

        self.write_to_output(out_text)

        if self.state_file_template and self.output_file:
            state_file = Path(self.state_file_template.format(epoch))
            state_text = (
                "**** writing the encoder parameters to " + state_file.as_posix()
            )

            self.write_to_output(state_text)

            torch.save(self.encoder.state_dict(), state_file)
