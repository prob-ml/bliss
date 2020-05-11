import sys
from abc import ABC, abstractmethod
import json

import h5py
import torch
from torch.utils.data import Dataset

from ..models import galaxy_net
from .. import utils

params_path = utils.data_path.joinpath("params_galaxy_datasets")


class SingleGalaxyDataset(Dataset, ABC):
    _params_file = None

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @classmethod
    def load_dataset_from_params(cls, params_file=None):
        """
        If not specified return the dataset from the default data_params file specified as a class attribute.
        """
        if params_file is None:
            params_file = cls._params_file
        assert (
            params_file is not None
        ), "Forgot to specify _params_file as class attribute"

        with open(params_file, "r") as fp:
            data_params = json.load(fp)
        return cls(**data_params)

    @abstractmethod
    def print_props(self, output=sys.stdout):
        # print relevant properties of the dataset, into given output stream.
        pass


class DecoderSamples(SingleGalaxyDataset):
    _params_file = params_path.joinpath("decoder_samples.json")

    def __init__(self, slen, decoder_file, num_bands=6, latent_dim=8, num_images=1000):
        """
        Load and sample from the specified decoder in `decoder_file`.

        :param slen: should match the ones loaded.
        :param latent_dim:
        :param num_images: Number of images to return when training in a network.
        :param num_bands:
        :param decoder_file: The file from which to load the `state_dict` of the decoder.
        :type decoder_file: Path object.
        """
        super().__init__()
        assert latent_dim == 8, "Not implemented any other decoder galaxy network"

        self.dec = galaxy_net.CenteredGalaxyDecoder(slen, latent_dim, num_bands).to(
            utils.device
        )
        self.dec.load_state_dict(torch.load(utils.data_path.joinpath(decoder_file)))
        self.num_bands = num_bands
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
    _params_file = params_path.joinpath("h5_cat.json")

    def __init__(self, h5_file, slen, num_bands):
        """
        A dataset created from single galaxy images in a h5py file.
        Args:
            h5_file: relative to data directory. 
            slen:
            num_bands:
        """
        super().__init__()
        h5_file_path = utils.data_path.joinpath(h5_file)

        self.file = h5py.File(h5_file_path, "r")
        assert utils.image_h5_name in self.file, "The dataset is not in this file"

        self.dset = self.file[utils.image_h5_name]
        self.num_bands = self.dset.shape[1]
        self.slen = self.dset.shape[2]
        assert (
            self.slen == slen == self.dset.shape[3]
        ), "slen does not match expected values."
        assert (
            self.num_bands == num_bands
        ), "Number of bands in training and in dataset do not match."

        assert utils.background_h5_name in self.dset.attrs, "Background is not in file"
        self.background = self.dset.attrs[utils.background_h5_name]

    def __len__(self):
        """
        Number of images saved in the file.
        :return:
        """
        return self.dset.shape[0]

    def __getitem__(self, idx):
        return {
            "image": self.dset[idx],  # shape = (num_bands, slen, slen)
            "background": self.background,
            "num_galaxies": 1,
        }

    def print_props(self, prop_file=sys.stdout):
        pass

    def __exit__(self):
        self.file.close()
