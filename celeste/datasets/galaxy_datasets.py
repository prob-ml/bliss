import sys
from abc import ABC, abstractmethod
import json

import h5py
import torch
from torch.utils.data import Dataset

from ..models import galaxy_net
from .. import device


class SingleGalaxyDataset(Dataset, ABC):
    _params_file = None

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @classmethod
    def load_dataset_from_params(cls, params_file):
        """
        If not specified return the dataset from the default data_params file specified as a
        class attribute.
        """

        with open(params_file, "r") as fp:
            data_params = json.load(fp)
        return cls(**data_params)

    @abstractmethod
    def print_props(self, output=sys.stdout):
        # print relevant properties of the dataset, into given output stream.
        pass


class DecoderSamples(SingleGalaxyDataset):
    def __init__(self, slen, decoder_file, n_bands=6, latent_dim=8, num_images=1000):
        """
        Load and sample from the specified decoder in `decoder_file`.

        :param slen: should match the ones loaded.
        :param latent_dim:
        :param num_images: Number of images to return when training in a network.
        :param n_bands:
        :param decoder_file: The file from which to load the `state_dict` of the decoder.
        :type decoder_file: Path object, full path.
        """
        super().__init__()
        assert latent_dim == 8, "Only implemented networks with latent_dim == 8"

        self.dec = galaxy_net.CenteredGalaxyDecoder(slen, latent_dim, n_bands).to(
            device
        )
        self.dec.load_state_dict(torch.load(decoder_file, map_location=device))
        self.n_bands = n_bands
        self.slen = slen
        self.num_images = num_images
        self.latent_dim = latent_dim

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        """
        Return numpy object.
        :param idx:
        :return: shape = (n_bands, slen, slen)
        """
        return self.dec.get_sample(1, return_latent=False).view(
            -1, self.slen, self.slen
        )

    def get_batch(self, batchsize):
        # returns = (z, images) where z.shape = (n_samples, latent_dim) and images.shape =
        # (n_samples, n_bands, slen, slen)

        return self.dec.get_sample(batchsize, return_latent=True)

    def print_props(self, output=sys.stdout):
        pass


class H5Catalog(Dataset):
    def __init__(self, h5_file, slen, n_bands):
        """
        A dataset created from single galaxy images in a h5py file.
        Args:
            h5_file: full path.
            slen:
            n_bands:
        """
        super().__init__()

        self.file = h5py.File(h5_file, "r")

        assert "images" in self.file, "The dataset is not in this file"

        self.dset = self.file["images"]
        self.n_bands = self.dset.shape[1]
        self.slen = self.dset.shape[2]
        assert (
            self.slen == slen == self.dset.shape[3]
        ), "slen does not match expected values."
        assert (
            self.n_bands == n_bands
        ), "Number of bands in training and in dataset do not match."

        assert "background" in self.dset.attrs, "Background is not in file"
        self.background = self.dset.attrs["background"]

    def __len__(self):
        """
        Number of images saved in the file.
        :return:
        """
        return self.dset.shape[0]

    def __getitem__(self, idx):
        return {
            "image": self.dset[idx],  # shape = (n_bands, slen, slen)
            "background": self.background,
            "num_galaxies": 1,
        }

    def print_props(self, prop_file=sys.stdout):
        pass

    def __exit__(self):
        self.file.close()
