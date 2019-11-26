import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import datasets
import color_net
import numpy as np


class TrainColor(object):
    def __init__(self, epochs=1000, batch_size=100, latent_dim=12, num_bands=6, dir_name=None):
        self.ds = datasets.ColorData()
        self.vae = color_net.ColorNet(latent_dim=latent_dim, num_bands=num_bands)

        tt_split = int(0.1 * len(self.ds))  # len(ds) = number of images?
        test_indices = np.mgrid[:tt_split]  # 10% of data only is for test.
        train_indices = np.mgrid[tt_split:len(self.ds)]

        self.optimizer = Adam(self.vae.parameters(), lr=lr, weight_decay=1e-6)
        self.test_loader = DataLoader(self.ds, batch_size=batch_size,
                                      num_workers=2, pin_memory=True,
                                      sampler=SubsetRandomSampler(test_indices))
        self.train_loader = DataLoader(self.ds, batch_size=batch_size,
                                       num_workers=2, pin_memory=True,
                                       sampler=SubsetRandomSampler(train_indices))

        self.dir_name = dir_name  # where to save results.

    def save_props(self):
        pass

    def evaluate_and_log(self, epoch):
        params = Path(self.dir_name, 'params')
        params.mkdir(parents=True, exist_ok=True)
        vae_file = params.joinpath("vae_params_{}.dat".format(epoch))
        loss_file = Path(args.dir, "loss.txt")
        print("  * writing the network's parameters to disk...")
        torch.save(self.vae.state_dict(), vae_file.as_posix())

        print("  * evaluating test loss...")
        test_loss = self.eval_epoch()
        print("  * test loss: {:.0f}\n".format(test_loss))

        with open(loss_file.as_posix(), 'a') as f:
            f.write(f"epoch {epoch}, test loss: {test_loss}\n")

    def train_epoch(self, data):
        self.vae.train()
        avg_loss = 0.0

        for batch_idx, color_sample in enumerate(self.train_loader):
            loss = self.vae.loss(color_sample)
            avg_loss += loss.item()

            self.optimizer.zero_grad()  # clears the gradients of all optimized torch.Tensors
            loss.backward()  # propagate this loss in the network.
            self.optimizer.step()  # only part where weights are changed.

        avg_loss /= len(self.train_loader.sampler)

    def eval_epoch(self, data):
        self.vae.eval()  # set in evaluation mode = no need to compute gradients or allocate memory for them.
        avg_loss = 0.0

        with torch.no_grad():  # no need to compute gradients outside training.
            for batch_idx, data in enumerate(self.test_loader):
                loss = self.vae.loss(data)
                avg_loss += loss.item()  # gets number from tensor containing single value.

        avg_loss /= len(self.test_loader.sampler)
        return avg_loss
