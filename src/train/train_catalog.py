import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import numpy as np
from pathlib import Path
import inspect

from GalaxyModel.src.data import catalog_datasets
from GalaxyModel.src.models import catalog_net


class TrainCatalog(object):

    def __init__(self, epochs=1000, batch_size=64, latent_dim: int = 10, training_examples=100, evaluation_examples=10,
                 dir_name=None):

        raise NotImplementedError("Should not be used until it is brought up to date with the other"
                                  "train files.")

        self.ds = catalog_datasets.CatsimData()
        self.vae = catalog_net.catalogNet(latent_dim=latent_dim, num_params=self.ds.num_params)
        self.epochs = epochs
        self.batch_size = batch_size
        self.training_examples = training_examples
        self.evaluation_examples = evaluation_examples
        self.lr = 1e-4

        tt_split = int(0.1 * len(self.ds))
        test_indices = np.mgrid[:tt_split]  # 10% of data only is for test.
        train_indices = np.mgrid[tt_split:len(self.ds)]

        self.optimizer = Adam(self.vae.parameters(), lr=self.lr, weight_decay=1e-6)
        self.test_loader = DataLoader(self.ds, batch_size=self.batch_size,
                                      num_workers=2, pin_memory=True,
                                      sampler=SubsetRandomSampler(test_indices))
        self.train_loader = DataLoader(self.ds, batch_size=self.batch_size,
                                       num_workers=2, pin_memory=True,
                                       sampler=SubsetRandomSampler(train_indices))

        self.dir_name = dir_name  # where to save results.
        self.save_props()

    def save_props(self):
        pass

    def evaluate_and_log(self, epoch):
        params = Path(self.dir_name, 'params')
        params.mkdir(parents=True, exist_ok=True)

        print("  * writing the network's parameters to disk...")
        vae_file = params.joinpath("vae_params_{}.dat".format(epoch))
        torch.save(self.vae.state_dict(), vae_file.as_posix())

        loss_file = Path(self.dir_name, "loss.txt")
        print("  * evaluating test loss...")
        test_loss = self.eval_epoch()
        print("  * test loss: {:.0f}\n".format(test_loss))

        with open(loss_file.as_posix(), 'a') as f:
            f.write(f"epoch {epoch}, test loss: {test_loss}\n")

    def train_epoch(self):
        self.vae.train()
        avg_loss = 0.0

        for batch_idx, sample in enumerate(self.train_loader):
            loss = self.vae.loss(sample.cuda())
            avg_loss += loss.item()

            self.optimizer.zero_grad()  # clears the gradients of all optimized torch.Tensors
            loss.backward()  # propagate this loss in the network.
            self.optimizer.step()  # only part where weights are changed.

            if batch_idx > self.training_examples:  # break after enough examples have been taken.
                break

        avg_loss /= self.batch_size * self.training_examples
        return avg_loss

    def eval_epoch(self):
        self.vae.eval()  # set in evaluation mode = no need to compute gradients or allocate memory for them.
        avg_loss = 0.0

        with torch.no_grad():  # no need to compute gradients outside training.
            for batch_idx, sample in enumerate(self.test_loader):

                loss = self.vae.loss(sample.cuda())
                avg_loss += loss.item()  # gets number from tensor containing single value.

                if batch_idx > self.evaluation_examples:  # break after enough examples have been taken.
                    break

        avg_loss /= self.batch_size * self.evaluation_examples
        return avg_loss

    @classmethod
    def add_args(cls, parser):
        raise NotImplementedError("Not implemented this yet.")

    @classmethod
    def from_args(cls, args_dict):
        parameters = inspect.signature(cls).parameters
        filtered_dict = {param: value for param, value in args_dict.items() if param in parameters}
        return cls(**filtered_dict)
