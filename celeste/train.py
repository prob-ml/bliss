from pathlib import Path
from abc import ABC, abstractmethod
import warnings
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from . import utils
from . import sleep


def set_seed(seed):
    np.random.seed(99999)
    _ = torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainModel(ABC):
    def __init__(
        self,
        model,
        dataset,
        slen,
        num_bands,
        lr=1e-4,
        weight_decay=1e-6,
        batchsize=64,
        eval_every=10,
        out_name=None,
        dloader_params=None,
        seed=42,
        verbose=False,
    ):
        set_seed(seed)  # seed for training.

        self.dataset = dataset
        self.slen = slen
        self.num_bands = num_bands
        self.batchsize = batchsize

        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        # evaluation and results
        self.verbose = verbose
        self.eval_every = eval_every
        self.out_name = out_name
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
            size_train = len(self.dataset) - self.size_test
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
        out_dir = utils.results_path.joinpath(self.out_name)
        if out_dir.exists():
            warnings.warn(
                "The output directory already exists, overwriting previous results."
            )
        out_dir.mkdir(exist_ok=True, parents=True)

        state_file_template = out_dir.joinpath(
            "state_{}.dat"
        ).as_posix()  # insert epoch later.

        output_file = out_dir.joinpath(
            "output.txt"
        )  # save the output that is being printed.

        if self.verbose:
            print(f"output file: {output_file.as_posix()}")
            print(f"state file format: {state_file_template}")

        return output_file, state_file_template

    def run(self, n_epochs):
        for epoch in range(n_epochs):
            self.step(epoch, train=True)

            if epoch % self.eval_every == 0:
                self.step(epoch, train=False)

    def step(self, epoch, train=True):
        # train or evaluate in the next epoch
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        t0 = time.time()
        batch_generator = self.get_batch_generator()
        avg_results = None

        for batch in batch_generator:
            results = self.get_results(batch)
            loss = self.get_loss(results)
            avg_results = self.update_avg(avg_results, results)

            if train:
                loss.backward()
                self.optimizer.step()

        if train:
            self.log_train(epoch, avg_results, t0)
        else:
            self.log_eval(epoch, avg_results)

    @abstractmethod
    def get_batch_generator(self):
        # return an iterator over all batches in a single epoch (if using data loader, then
        # this would be the dloader)
        pass

    @abstractmethod
    def get_results(self, batch):
        # get all training/evaluation results from a batch, this includes loss and maybe other useful things to log.
        pass

    # TODO: A bit clunky in my opinion, open to suggestions.
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
    def __init__(self, *args, n_source_params, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = self.model
        self.n_source_params = n_source_params

    # TODO: A bit hacky, but ok for now since we will move to a combined dataset soon.
    #  also avoids adding annoying flag of galaxy or star.
    def _get_params_from_batch(self, batch):
        class_name = self.dataset.__class__.__name__
        if class_name == "GalaxyDataset":
            return batch["gal_params"], batch["locs"], batch["images"]

        elif class_name == "StarDataset":
            return batch["log_fluxes"], batch["locs"], batch["images"]

        else:
            raise ValueError("Added an incompatible dataset.")

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
        true_source_params, true_locs, images = self._get_params_from_batch(batch)

        # evaluate log q
        loss, counter_loss, locs_loss, source_params_loss = sleep.get_inv_kl_loss(
            self.encoder, images, true_locs, true_source_params
        )[0:4]

        return loss, counter_loss, locs_loss, source_params_loss

    def update_avg(self, avg_results, results):
        loss, counter_loss, locs_loss, source_params_loss = results
        avg_loss, avg_counter_loss, avg_locs_loss, avg_source_params_loss = avg_results

        avg_loss += loss.item() * self.batchsize / len(self.dataset)
        avg_counter_loss += counter_loss.sum().item() / (
            len(self.dataset) * self.encoder.n_tiles
        )
        avg_source_params_loss += source_params_loss.sum().item() / (
            len(self.dataset) * self.encoder.n_tiles
        )
        avg_locs_loss += locs_loss.sum().item() / (
            len(self.dataset) * self.encoder.n_tiles
        )

        return avg_loss, avg_counter_loss, avg_locs_loss, avg_source_params_loss

    def log_train(
        self, epoch, avg_results, t0,
    ):
        avg_loss, counter_loss, locs_loss, source_param_loss = avg_results
        # print and save test results.
        elapsed = time.time() - t0
        out_text = (
            f"{epoch} loss: {avg_loss:.4f}; counter loss: {counter_loss:.4f}; locs loss: {locs_loss:.4f}; "
            f"source_params loss: {source_param_loss:.4f} \t [{elapsed:.1f} seconds]"
        )

        with open(self.output_file, "a") as out:
            print(out_text, file=out)

    def log_eval(self, epoch, avg_results):
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
        print(out_text)
        with open(self.output_file, "a") as out:
            print(out_text, file=out)

        state_file = Path(self.state_file_template.format(epoch))
        print("writing the encoder parameters to " + state_file.as_posix())
        torch.save(self.encoder.state_dict(), state_file)


# import matplotlib.pyplot as plt
# from .models import galaxy_net
# class TrainSingleGalaxy(TrainModel):
#     def __init__(
#         self,
#         dataset,
#         slen,
#         num_bands,
#         num_workers=None,
#         epochs=None,
#         batch_size=None,
#         eval_every=None,
#         dir_name=None,
#     ):
#         """
#         This function now iterates through the whole dataset in each epoch, with the number of objects in each epoch
#         depending on the __len__ attribute of the dataset.
#         """
#
#         # constants for optimization and network.
#         self.latent_dim = 8
#
#         self.dataset = dataset
#
#         self.vae = galaxy_net.OneCenteredGalaxy(
#             slen, num_bands=num_bands, latent_dim=self.latent_dim
#         )
#
#         self.dir_name = dir_name  # where to save results.
#         self.save_props()  # save relevant properties to a file so we know how to reconstruct these results.
#
#     def save_props(self):
#         prop_file = open(f"{self.dir_name}/props.txt", "w")
#         print(
#             f"dataset: {self.dataset.__class__.__name__}\n"
#             f"dataset size: {len(self.dataset)}\n"
#             f"epochs: {self.epochs}\n"
#             f"batch_size: {self.batch_size}\n"
#             f"eval_every: {self.eval_every}\n"
#             f"learning rate: {self.lr}\n"
#             f"slen: {self.slen}\n"
#             f"latent dim: {self.vae.latent_dim}\n",
#             f"num bands: {self.num_bands}\n",
#             f"num workers: {self.num_workers}\n",
#             file=prop_file,
#         )
#         self.dataset.print_props(prop_file)
#         prop_file.close()
#
#     # @profile
#     def train_epoch(self):
#         self.vae.train()
#         avg_loss = 0.0
#
#         for batch_idx, data in enumerate(self.train_loader):
#             image = data["image"].to(
#                 utils.device
#             )  # shape: [nsamples, num_bands, slen, slen]
#             background = data["background"].to(utils.device)
#
#             loss = self.vae.loss(image, background)
#             avg_loss += loss.item()
#
#             self.optimizer.zero_grad()  # clears the gradients of all optimized torch.Tensors
#             loss.backward()  # propagate this loss in the network.
#             self.optimizer.step()  # only part where weights are changed.
#
#         avg_loss /= self.size_train
#
#         return avg_loss
#
#     def evaluate_and_log(self, epoch):
#         """
#         Plot reconstructions and evaluate test loss. Also save vae and decoder for future use.
#         :param epoch:
#         :return:
#         """
#         print("  * plotting reconstructions...")
#         self.plot_reconstruction(epoch)
#
#         # paths
#         params = Path(self.dir_name, "params")
#         params.mkdir(parents=True, exist_ok=True)
#         loss_file = Path(self.dir_name, "loss.txt")
#         vae_file = params.joinpath("vae_params_{}.dat".format(epoch))
#         decoder_file = params.joinpath("decoder_params_{}.dat".format(epoch))
#
#         # write into files.
#         print("  * writing the network's parameters to disk...")
#         torch.save(self.vae.state_dict(), vae_file.as_posix())
#         torch.save(self.vae.dec.state_dict(), decoder_file.as_posix())
#
#         print("  * evaluating test loss...")
#         test_loss, avg_rmse = self.eval_epoch()
#         print("  * test loss: {:.0f}".format(test_loss))
#         print(f" * avg_rmse: {avg_rmse}")
#
#         with open(loss_file.as_posix(), "a") as f:
#             f.write(f"epoch {epoch}, test loss: {test_loss}, avg rmse: {avg_rmse} \n")
#
#     def eval_epoch(self):
#         self.vae.eval()  # set in evaluation mode = no need to compute gradients or allocate memory for them.
#         avg_loss = 0.0
#         avg_rmse = 0.0
#
#         with torch.no_grad():  # no need to compute gradients outside training.
#             for batch_idx, data in enumerate(self.test_loader):
#                 image = data["image"].cuda()  # shape: [nsamples, num_bands, slen, slen]
#                 background = data["background"].cuda()
#
#                 loss = self.vae.loss(image, background)
#                 avg_loss += (
#                     loss.item()
#                 )  # gets number from tensor containing single value.
#                 avg_rmse += self.vae.rmse_pp(image, background).item()
#
#         avg_loss /= self.size_test
#         avg_rmse /= self.size_test
#         return avg_loss, avg_rmse
#
#     def plot_reconstruction(self, epoch):
#         """
#         Now for each epoch to evaluate, it creates a new folder with the reconstructions that include each of the bands.
#         :param epoch:
#         :return:
#         """
#         num_examples = min(10, self.batch_size)
#
#         num_cols = 3  # also look at recon_var
#
#         plots_path = Path(self.dir_name, f"plots")
#         bands_indices = [
#             min(2, self.num_bands - 1)
#         ]  # only i band if available, otherwise the highest band.
#         plots_path.mkdir(parents=True, exist_ok=True)
#
#         plt.ioff()
#         with torch.no_grad():
#             for batch_idx, data in enumerate(self.test_loader):
#                 image = data["image"].cuda()  # copies from cpu to gpu memory.
#                 background = data[
#                     "background"
#                 ].cuda()  # not having background will be a problem.
#                 num_galaxies = data["num_galaxies"]
#                 self.vae.eval()
#
#                 recon_mean, recon_var, _ = self.vae(image, background)
#                 for j in bands_indices:
#                     plt.figure(figsize=(5 * 3, 2 + 4 * num_examples))
#                     plt.tight_layout()
#                     plt.suptitle("Epoch {:d}".format(epoch))
#
#                     for i in range(num_examples):
#                         vmax1 = image[
#                             i, j
#                         ].max()  # we are looking at the ith sample in the jth band.
#                         vmax2 = max(
#                             image[i, j].max(),
#                             recon_mean[i, j].max(),
#                             recon_var[i, j].max(),
#                         )
#
#                         plt.subplot(num_examples, num_cols, num_cols * i + 1)
#                         plt.title("image [{} galaxies]".format(num_galaxies[i]))
#                         plt.imshow(image[i, j].data.cpu().numpy(), vmax=vmax1)
#                         plt.colorbar()
#
#                         plt.subplot(num_examples, num_cols, num_cols * i + 2)
#                         plt.title("recon_mean")
#                         plt.imshow(recon_mean[i, j].data.cpu().numpy(), vmax=vmax1)
#                         plt.colorbar()
#
#                         plt.subplot(num_examples, num_cols, num_cols * i + 3)
#                         plt.title("recon_var")
#                         plt.imshow(recon_var[i, j].data.cpu().numpy(), vmax=vmax2)
#                         plt.colorbar()
#
#                     plot_file = plots_path.joinpath(f"plot_{epoch}_{j}")
#                     plt.savefig(plot_file.as_posix())
#                     plt.close()
#
#                 break
