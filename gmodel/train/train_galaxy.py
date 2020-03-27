import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import inspect

from ..utils import const
from ..data import galaxy_datasets
from ..models import galaxy_net


class TrainGalaxy(object):

    def __init__(self, slen: int = 40, num_bands: int = 1, num_workers: int = 2, h5_file: str = '',
                 fixed_size: const.str_bool = False, reconstruct_one: const.str_bool = True, epochs=None,
                 batch_size=None, evaluate=None, dir_name=None, dataset=None):
        """
        This function now iterates through the whole dataset in each epoch, with the number of objects in each epoch
        depending on the __len__ attribute of the dataset.
        :param slen:
        :param num_bands:
        :param num_workers:
        :param h5_file:
        :param fixed_size:
        :param reconstruct_one: Whether to plot all bands or only the 'i' band. [Default: True]
        :param epochs:
        :param batch_size:
        :param evaluate:
        :param dir_name:
        :param dataset:
        """
        assert num_workers == 0 or not dataset == "h5_catalog", "Num of workers should be 0 when not generating " \
                                                                "galaxies on the fly."

        # constants for optimization and network.
        self.latent_dim = 8
        self.lr = 1e-4
        self.weight_decay = 1e-6

        self.dataset = dataset
        self.slen = slen
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluate = evaluate
        self.num_bands = num_bands
        self.num_workers = num_workers
        self.fixed_size = fixed_size
        self.reconstruct_one = reconstruct_one

        self.ds = galaxy_datasets.decide_dataset(self.dataset, self.slen, self.num_bands, fixed_size=self.fixed_size,
                                                 h5_file=h5_file)
        assert len(self.ds) >= 1000, "Dataset is too small."

        self.vae = galaxy_net.OneCenteredGalaxy(self.slen, num_bands=self.num_bands, latent_dim=self.latent_dim)

        split = 0.1
        self.size_test = int(split * len(self.ds))
        self.size_train = len(self.ds) - self.size_test

        tt_split = int(split * len(self.ds))
        test_indices = np.mgrid[:tt_split]  # 10% of data only is for test.
        train_indices = np.mgrid[tt_split:len(self.ds)]

        self.optimizer = Adam(self.vae.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.evaluate:
            self.test_loader = DataLoader(self.ds, batch_size=self.batch_size,
                                          num_workers=self.num_workers, pin_memory=True,
                                          sampler=SubsetRandomSampler(test_indices))

        self.train_loader = DataLoader(self.ds, batch_size=self.batch_size,
                                       num_workers=self.num_workers, pin_memory=True,
                                       sampler=SubsetRandomSampler(train_indices))

        self.dir_name = dir_name  # where to save results.
        self.save_props()  # save relevant properties to a file so we know how to reconstruct these results.

    def save_props(self):
        # TODO: Create a directory file for easy look-up once we start producing a lot of these.
        prop_file = open(f"{self.dir_name}/props.txt", 'w')
        print(f"dataset: {self.dataset}\n"
              f"dataset size: {len(self.ds)}\n"
              f"epochs: {self.epochs}\n"
              f"batch_size: {self.batch_size}\n"
              f"evaluate: {self.evaluate}\n"
              f"learning rate: {self.lr}\n"
              f"slen: {self.slen}\n"
              f"latent dim: {self.vae.latent_dim}\n",
              f"num bands: {self.num_bands}\n",
              f"num workers: {self.num_workers}\n",
              file=prop_file)
        self.ds.print_props(prop_file)
        prop_file.close()

    # @profile
    def train_epoch(self):
        self.vae.train()
        avg_loss = 0.0

        for batch_idx, data in enumerate(self.train_loader):

            image = data["image"].cuda()  # shape: [nsamples, num_bands, slen, slen]
            background = data["background"].cuda()

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
        params = Path(self.dir_name, 'params')
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

        with open(loss_file.as_posix(), 'a') as f:
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
                avg_loss += loss.item()  # gets number from tensor containing single value.
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

        plots_path = Path(self.dir_name, f'plots')
        if self.reconstruct_one:
            bands_indices = [min(2, self.num_bands - 1)]  # only i band if available, otherwise the highest band.
        else:
            bands_indices = range(self.num_bands)
            plots_path = plots_path.joinpath(f"epoch_{epoch}")
        plots_path.mkdir(parents=True, exist_ok=True)

        plt.ioff()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                image = data["image"].cuda()  # copies from cpu to gpu memory.
                background = data["background"].cuda()  # not having background will be a problem.
                num_galaxies = data["num_galaxies"]
                self.vae.eval()

                recon_mean, recon_var, _ = self.vae(image, background)
                for j in bands_indices:
                    plt.figure(figsize=(5 * 3, 2 + 4 * num_examples))
                    plt.tight_layout()
                    plt.suptitle("Epoch {:d}".format(epoch))

                    for i in range(num_examples):
                        vmax1 = image[i, j].max()  # we are looking at the ith sample in the jth band.
                        vmax2 = max(image[i, j].max(), recon_mean[i, j].max(), recon_var[i, j].max())

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

    @classmethod
    def add_args(cls, parser):
        parameters = inspect.signature(cls).parameters
        for param in parameters:
            if param not in const.general_args and param != 'self':
                arg_form = const.to_argparse_form(param)
                parser.add_argument(arg_form, type=parameters[param].annotation, default=parameters[param].default,
                                    help='A parameter.')

    @classmethod
    def from_args(cls, args_dict):
        parameters = inspect.signature(cls).parameters
        filtered_dict = {param: value for param, value in args_dict.items() if param in parameters}
        return cls(**filtered_dict)
