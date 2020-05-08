import numpy as np
import sys
import random
from abc import ABC, abstractmethod

from astropy.table import Table
import h5py
import torch
from torch.utils.data import Dataset
import descwl
import json

from ..models import galaxy_net
from ..utils import const
from . import draw_catsim

params_path = const.data_path.joinpath("params_galaxy_datasets")


class GalaxyDataset(Dataset, ABC):
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


class DecoderSamples(GalaxyDataset):
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
            const.device
        )
        self.dec.load_state_dict(torch.load(const.data_path.joinpath(decoder_file)))
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


class H5Catalog(GalaxyDataset):
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
        h5_file_path = const.data_path.joinpath(h5_file)

        self.file = h5py.File(h5_file_path, "r")
        assert const.image_h5_name in self.file, "The dataset is not in this file"

        self.dset = self.file[const.image_h5_name]
        self.num_bands = self.dset.shape[1]
        self.slen = self.dset.shape[2]
        assert (
            self.slen == slen == self.dset.shape[3]
        ), "slen does not match expected values."
        assert (
            self.num_bands == num_bands
        ), "Number of bands in training and in dataset do not match."

        assert const.background_h5_name in self.dset.attrs, "Background is not in file"
        self.background = self.dset.attrs[const.background_h5_name]

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

    def print_props(self, prop_file):
        pass

    @classmethod
    def load_dataset_from_params(cls, params_file=None):
        raise NotImplementedError("Need to make params_file for this class.")

    def __exit__(self):
        self.file.close()


class CatsimGalaxies(GalaxyDataset):
    _params_file = params_path.joinpath("catsim_single_band.json")

    def __init__(
        self,
        survey_name=None,
        slen=41,
        filter_dict=None,
        snr=200,
        num_bands=1,
        bands=None,
        dtype=np.float32,
        catalog_file="OneDegSq.fits",
        preserve_flux=False,
        add_noise=True,
        render_kwargs=None,
    ):
        """
        This class reads a random entry from the OneDegSq.fits file (sample from the Catsim catalog) and returns a
        galaxy drawn from the catalog with realistic seeing conditions using functions from WeakLensingDeblending.

        For now, only one galaxy can be returned at once.

        :param snr: The SNR of the galaxy to draw, if None uses the actually seeing SNR from LSST survey.
        :param render_kwargs: Additional keyword arguments that will go into the renderer.
        :param filter_dict: Exclude some entries from based CATSIM on dict of filters, default is to exclude >=25.3 i_ab
        """
        super().__init__()
        assert survey_name == "LSST", "Only using default survey name for now is LSST."
        assert num_bands in [1, 6], "Only 1 or 6 bands are supported."
        assert (
            slen >= 41
        ), "Does not seem to work well if the number of pixels is too low."
        assert slen % 2 == 1, "Odd number of pixels is preferred."
        assert (
            filter_dict is None
        ), "Not supporting different dict yet, need to change argparse + save_props below."
        assert dtype is np.float32, "Only float32 is supported for now."
        assert (
            preserve_flux is False
        ), "Otherwise variance of the noise will change which is not desirable."
        # ToDo: Create a test or assertion to check that mean == variance approx.
        assert num_bands == len(bands)

        self.survey_name = survey_name
        self.bands = bands
        self.num_bands = num_bands

        self.slen = slen
        self.pixel_scale = descwl.survey.Survey.get_defaults(self.survey_name, "*")[
            "pixel_scale"
        ]
        self.stamp_size = self.pixel_scale * self.slen  # arcsecs.
        self.snr = snr
        self.dtype = dtype
        self.preserve_flux = preserve_flux
        self.add_noise = add_noise

        self.renderer = draw_catsim.CatsimRenderer(
            self.survey_name,
            self.bands,
            self.stamp_size,
            self.pixel_scale,
            snr=self.snr,
            dtype=self.dtype,
            preserve_flux=self.preserve_flux,
            add_noise=self.add_noise,
            **render_kwargs,
        )
        self.background = self.renderer.background

        # prepare catalog table.
        self.table = Table.read(const.data_path.joinpath(catalog_file))
        self.table = self.table[
            np.random.permutation(len(self.table))
        ]  # shuffle in case that order matters.
        self.filtered_dict = (
            CatsimGalaxies.get_default_filters() if filter_dict is None else filter_dict
        )
        self.cat = self.get_filtered_table()

    def __len__(self):
        return len(self.cat)

    # ToDo: Remove all non-visible sources from catalogue directly?
    def __getitem__(self, idx):

        while True:  # loop until visible galaxy is selected.
            try:
                entry = self.cat[idx]
                final, background = self.renderer.draw(entry)
                break

            except descwl.render.SourceNotVisible:
                idx = np.random.choice(
                    np.arange(len(self))
                )  # select some other random galaxy to return.

        return {"image": final, "background": background, "num_galaxies": 1}

    def print_props(self, output=sys.stdout):
        print(
            f"slen: {self.slen} \n"
            f"snr: {self.snr} \n"
            f"survey name: {self.survey_name} \n"
            f"bands: {self.bands}\n"
            f"pixel scale: {self.pixel_scale}\n"
            f"filter_dict: Default\n"
            f"min_snr: {self.renderer.min_snr}\n"
            f"truncate_radius: {self.renderer.truncate_radius}\n"
            f"add_noise: {self.renderer.add_noise}\n"
            f"preserve_flux: {self.renderer.preserve_flux}\n"
            f"dtype: {self.renderer.dtype}",
            file=output,
        )

    def get_filtered_table(self):
        cat = self.table.copy()
        for param, pfilter in self.filtered_dict.items():
            cat = cat[pfilter(param, cat)]
        return cat

    @staticmethod
    def get_default_filters():
        # ToDo: Make a cut on the size? Something like:
        # sizes = self.renderer.get_size(self.cat); cat = cat[sizes < 30] (size in pixels)
        filters = dict(
            i_ab=lambda param, x: x[param]
            <= 25.3  # cut on magnitude same as BTK does (gold sample)
        )
        return filters


def generate_images(
    dataset_cls, dataset_kwargs, out_path, prop_file_path=None, num_images=1,
):
    """
    Generate images from dataset cls and save num_images into h5py file.
    """

    ds = dataset_cls(**dataset_kwargs)

    if prop_file_path:
        with open(prop_file_path, "w") as prop_file:
            ds.print_props(prop_file)

    with h5py.File(out_path, "w") as images_file:
        hds_shape = (num_images, ds.num_bands, ds.slen, ds.slen)
        hds = images_file.create_dataset(const.image_h5_name, hds_shape, dtype=ds.dtype)
        for i in range(num_images):
            random_idx = random.randrange(len(ds))
            output = ds[random_idx]
            image = output["image"]
            hds[i, :, :, :] = image
            hds.flush()
        hds.attrs[const.background_h5_name] = ds.background
        hds.flush()
