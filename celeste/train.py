from pathlib import Path
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .models import galaxy_net
from . import utils


class TrainGalaxy(object):
    def __init__(
        self,
        dataset,
        slen,
        num_bands,
        num_workers=None,
        epochs=None,
        batch_size=None,
        eval_every=None,
        dir_name=None,
    ):
        """
        This function now iterates through the whole dataset in each epoch, with the number of objects in each epoch
        depending on the __len__ attribute of the dataset.
        """

        # constants for optimization and network.
        self.latent_dim = 8
        self.lr = 1e-4
        self.weight_decay = 1e-6

        self.slen = slen
        self.num_bands = num_bands
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.num_workers = num_workers

        self.dataset = dataset
        assert len(self.dataset) >= 1000, "Dataset is too small."

        self.vae = galaxy_net.OneCenteredGalaxy(
            slen, num_bands=num_bands, latent_dim=self.latent_dim
        )

        split = 0.1
        self.size_test = int(split * len(self.dataset))
        self.size_train = len(self.dataset) - self.size_test

        tt_split = int(split * len(self.dataset))
        test_indices = np.mgrid[:tt_split]  # 10% of data only is for test.
        train_indices = np.mgrid[tt_split : len(self.dataset)]

        self.optimizer = Adam(
            self.vae.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.eval_every:
            self.test_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=SubsetRandomSampler(test_indices),
            )

        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=SubsetRandomSampler(train_indices),
        )

        self.dir_name = dir_name  # where to save results.
        self.save_props()  # save relevant properties to a file so we know how to reconstruct these results.

    def save_props(self):
        prop_file = open(f"{self.dir_name}/props.txt", "w")
        print(
            f"dataset: {self.dataset.__class__.__name__}\n"
            f"dataset size: {len(self.dataset)}\n"
            f"epochs: {self.epochs}\n"
            f"batch_size: {self.batch_size}\n"
            f"eval_every: {self.eval_every}\n"
            f"learning rate: {self.lr}\n"
            f"slen: {self.slen}\n"
            f"latent dim: {self.vae.latent_dim}\n",
            f"num bands: {self.num_bands}\n",
            f"num workers: {self.num_workers}\n",
            file=prop_file,
        )
        self.dataset.print_props(prop_file)
        prop_file.close()

    # @profile
    def train_epoch(self):
        self.vae.train()
        avg_loss = 0.0

        for batch_idx, data in enumerate(self.train_loader):
            image = data["image"].to(
                utils.device
            )  # shape: [nsamples, num_bands, slen, slen]
            background = data["background"].to(utils.device)

            loss = self.vae.loss(image, background)
            avg_loss += loss.item()

            self.optimizer.zero_grad()  # clears the gradients of all optimized torch.Tensors
            loss.backward()  # propagate this loss in the network.
            self.optimizer.step()  # only part where weights are changed.

        avg_loss /= self.size_train

        return avg_loss

    def evaluate_and_log(self, epoch):
        """
        Plot reconstructions and evaluate test loss. Also save vae and decoder for future use.
        :param epoch:
        :return:
        """
        print("  * plotting reconstructions...")
        self.plot_reconstruction(epoch)

        # paths
        params = Path(self.dir_name, "params")
        params.mkdir(parents=True, exist_ok=True)
        loss_file = Path(self.dir_name, "loss.txt")
        vae_file = params.joinpath("vae_params_{}.dat".format(epoch))
        decoder_file = params.joinpath("decoder_params_{}.dat".format(epoch))

        # write into files.
        print("  * writing the network's parameters to disk...")
        torch.save(self.vae.state_dict(), vae_file.as_posix())
        torch.save(self.vae.dec.state_dict(), decoder_file.as_posix())

        print("  * evaluating test loss...")
        test_loss, avg_rmse = self.eval_epoch()
        print("  * test loss: {:.0f}".format(test_loss))
        print(f" * avg_rmse: {avg_rmse}")

        with open(loss_file.as_posix(), "a") as f:
            f.write(f"epoch {epoch}, test loss: {test_loss}, avg rmse: {avg_rmse} \n")

    def eval_epoch(self):
        self.vae.eval()  # set in evaluation mode = no need to compute gradients or allocate memory for them.
        avg_loss = 0.0
        avg_rmse = 0.0

        with torch.no_grad():  # no need to compute gradients outside training.
            for batch_idx, data in enumerate(self.test_loader):
                image = data["image"].cuda()  # shape: [nsamples, num_bands, slen, slen]
                background = data["background"].cuda()

                loss = self.vae.loss(image, background)
                avg_loss += (
                    loss.item()
                )  # gets number from tensor containing single value.
                avg_rmse += self.vae.rmse_pp(image, background).item()

        avg_loss /= self.size_test
        avg_rmse /= self.size_test
        return avg_loss, avg_rmse

    def plot_reconstruction(self, epoch):
        """
        Now for each epoch to evaluate, it creates a new folder with the reconstructions that include each of the bands.
        :param epoch:
        :return:
        """
        num_examples = min(10, self.batch_size)

        num_cols = 3  # also look at recon_var

        plots_path = Path(self.dir_name, f"plots")
        bands_indices = [
            min(2, self.num_bands - 1)
        ]  # only i band if available, otherwise the highest band.
        plots_path.mkdir(parents=True, exist_ok=True)

        plt.ioff()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                image = data["image"].cuda()  # copies from cpu to gpu memory.
                background = data[
                    "background"
                ].cuda()  # not having background will be a problem.
                num_galaxies = data["num_galaxies"]
                self.vae.eval()

                recon_mean, recon_var, _ = self.vae(image, background)
                for j in bands_indices:
                    plt.figure(figsize=(5 * 3, 2 + 4 * num_examples))
                    plt.tight_layout()
                    plt.suptitle("Epoch {:d}".format(epoch))

                    for i in range(num_examples):
                        vmax1 = image[
                            i, j
                        ].max()  # we are looking at the ith sample in the jth band.
                        vmax2 = max(
                            image[i, j].max(),
                            recon_mean[i, j].max(),
                            recon_var[i, j].max(),
                        )

                        plt.subplot(num_examples, num_cols, num_cols * i + 1)
                        plt.title("image [{} galaxies]".format(num_galaxies[i]))
                        plt.imshow(image[i, j].data.cpu().numpy(), vmax=vmax1)
                        plt.colorbar()

                        plt.subplot(num_examples, num_cols, num_cols * i + 2)
                        plt.title("recon_mean")
                        plt.imshow(recon_mean[i, j].data.cpu().numpy(), vmax=vmax1)
                        plt.colorbar()

                        plt.subplot(num_examples, num_cols, num_cols * i + 3)
                        plt.title("recon_var")
                        plt.imshow(recon_var[i, j].data.cpu().numpy(), vmax=vmax2)
                        plt.colorbar()

                    plot_file = plots_path.joinpath(f"plot_{epoch}_{j}")
                    plt.savefig(plot_file.as_posix())
                    plt.close()

                break
