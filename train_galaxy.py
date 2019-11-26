import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import datasets
import galaxy_net
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import inspect
import utils


class TrainGalaxy(object):

    def __init__(self, slen: int = 30, epochs: int = 1000, batch_size: int = 100, num_examples=1000,
                 num_bands: int = 1, dir_name: str = None, dataset: str = 'galbasic'):

        self.dataset = dataset
        self.slen = slen
        self.num_bands = num_bands
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.lr = 1e-6
        self.decide_dataset()

        self.vae = galaxy_net.OneCenteredGalaxy(self.slen, num_bands=self.num_bands, latent_dim=8)

        tt_split = int(0.1 * len(self.ds))  # len(ds) = number of images?
        test_indices = np.mgrid[:tt_split]  # 10% of data only is for test.
        train_indices = np.mgrid[tt_split:len(self.ds)]

        self.optimizer = Adam(self.vae.parameters(), lr=self.lr, weight_decay=1e-6)
        self.test_loader = DataLoader(self.ds, batch_size=batch_size,
                                      num_workers=2, pin_memory=True,
                                      sampler=SubsetRandomSampler(test_indices))
        self.train_loader = DataLoader(self.ds, batch_size=batch_size,
                                       num_workers=2, pin_memory=True,
                                       sampler=SubsetRandomSampler(train_indices))

        self.dir_name = dir_name  # where to save results.
        self.save_props()

    def decide_dataset(self):
        if self.dataset == 'galbasic':
            self.ds = datasets.GalBasic(self.slen, min_galaxies=1, max_galaxies=1, mean_galaxies=1,
                                        num_images=self.num_examples,
                                        centered=True, sky=700)

        elif self.dataset == 'synthetic':  # default one.
            self.ds = datasets.Synthetic(self.slen, min_galaxies=1, max_galaxies=1, mean_galaxies=1,
                                         num_images=self.num_examples,
                                         centered=True, num_bands=self.num_bands)

        else:
            raise NotImplementedError("Not implemented that galaxy dataset yet.")

    def save_props(self):
        # TODO: Create a directory file for easy look-up once we start producing several of these.
        prop_file = open(f"{self.dir}/props.txt", 'w')
        print(f"dataset: {self.dataset} \n"
              f"epochs: {self.epochs} \n"
              f"batch_size: {self.batch_size}\n"
              f"learning rate: {self.lr}\n"
              f"sky level: {self.ds.sky}\n"
              f"slen: {self.slen}\n"
              f"snr: {self.ds.snr}\n"
              f"flux: {self.ds.flux}\n"
              f"num bands: {self.num_bands}", file=prop_file)
        prop_file.close()

    def evaluate_and_log(self, epoch):
        print("  * plotting reconstructions...")
        self.plot_reconstruction(epoch)

        # paths
        params = Path(self.dir_name, 'params')
        params.mkdir(parents=True, exist_ok=True)
        loss_file = Path(self.dir_name, "loss.txt")
        vae_file = params.joinpath("vae_params_{}.dat".format(epoch))

        # write into files.
        print("  * writing the network's parameters to disk...")
        torch.save(self.vae.state_dict(), vae_file.as_posix())

        print("  * evaluating test loss...")
        test_loss, avg_mse = self.eval_epoch()
        print("  * test loss: {:.0f}\n".format(test_loss))
        print(f" * avg_mse: {avg_mse}")

        with open(loss_file.as_posix(), 'a') as f:
            f.write(f"epoch {epoch}, test loss: {test_loss}, avg mse: {avg_mse}\n")

    def train_epoch(self):
        self.vae.train()
        avg_loss = 0.0

        for batch_idx, color_sample in enumerate(self.train_loader):
            loss = self.vae.loss(color_sample)
            avg_loss += loss.item()

            self.optimizer.zero_grad()  # clears the gradients of all optimized torch.Tensors
            loss.backward()  # propagate this loss in the network.
            self.optimizer.step()  # only part where weights are changed.

        avg_loss /= len(self.train_loader.sampler)
        return avg_loss

    def eval_epoch(self):
        self.vae.eval()  # set in evaluation mode = no need to compute gradients or allocate memory for them.
        avg_loss = 0.0
        avg_mse = 0.0

        with torch.no_grad():  # no need to compute gradients outside training.
            for batch_idx, data in enumerate(self.test_loader):
                image = data["image"].cuda()  # shape: [nsamples, num_bands, slen, slen]
                background = data["background"].cuda()
                loss = self.vae.loss(image, background)
                avg_loss += loss.item()  # gets number from tensor containing single value.
                mse = self.vae.mse(image, background)
                avg_mse += mse.item()

        avg_loss /= len(self.test_loader.sampler)
        avg_mse /= len(self.test_loader.sampler)
        return avg_loss, avg_mse

    def plot_reconstruction(self, epoch):
        num_examples = min(10, self.batch_size)

        plt.ioff()
        plt.figure(figsize=(5 * 3, 2 + 4 * num_examples))
        plt.tight_layout()
        plt.suptitle("Epoch {:d}".format(epoch))

        num_cols = 3  # also look at recon_var

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                image = data["image"].cuda()  # copies from cpu to gpu memory.
                background = data["background"].cuda()  # maybe not having background will be a problem.
                num_galaxies = data["num_galaxies"]
                self.vae.eval()

                recon_mean, recon_var, _ = self.vae(image, background)

                for i in range(num_examples):
                    vmax1 = image[i, 0].max()  # we are looking at the ith sample in the first band.
                    plt.subplot(num_examples, num_cols, num_cols * i + 1)
                    plt.title("image [{} galaxies]".format(num_galaxies[i]))
                    plt.imshow(image[i, 0].data.cpu().numpy(), vmax=vmax1)
                    plt.colorbar()

                    plt.subplot(num_examples, num_cols, num_cols * i + 2)
                    plt.title("recon_mean")
                    plt.imshow(recon_mean[i, 0].data.cpu().numpy(), vmax=vmax1)
                    plt.colorbar()

                    plt.subplot(num_examples, num_cols, num_cols * i + 3)
                    plt.title("recon_var")
                    plt.imshow(recon_var[i, 0].data.cpu().numpy(), vmax=vmax1)
                    plt.colorbar()

                break

        plots = Path(self.dir_name, 'plots')
        plots.mkdir(parents=True, exist_ok=True)
        plot_file = plots.joinpath("plot_epoch_{}".format(epoch))
        plt.savefig(plot_file.as_posix())
        plt.close()

    @classmethod
    def add_args(cls, parser, arguments_so_far):
        parameters = inspect.signature(cls).parameters
        for param in parameters:
            if param not in arguments_so_far and param != 'self':
                arg_form = utils.to_argparse_form(param)
                print(arg_form, parameters[param].annotation, parameters[param].default)
                parser.add_argument(arg_form, type=parameters[param].annotation, default=parameters[param].default,
                                    help='')

    @classmethod
    def from_args(cls, args):
        parameters = inspect.signature(cls).parameters
        args_dict = vars(args)
        filtered_dict = {param: value for param, value in args_dict.items() if param in parameters}
        return cls(**filtered_dict)
