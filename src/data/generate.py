import numpy as np
import h5py
import random

from src.data import galaxy_datasets
from src.utils import const


def generate_images(dataset_name, out_path, prop_file_path=None, num_images=1,
                    slen=40, num_bands=6, fixed_size=False):
    """

    :param dataset_name: The name of the dataset from :mod:`galaxy_datasets` that you want to use.
    :param num_images: number of images to save.
    :return:
    """
    assert num_images >= 1, "At least one image must be produced."

    ds = galaxy_datasets.decide_dataset(dataset_name, slen, num_bands, fixed_size=fixed_size)

    if prop_file_path:
        with open(prop_file_path, 'w') as prop_file:
            ds.print_props(prop_file)

    with h5py.File(out_path, 'w') as images_file:
        hds_shape = (num_images, ds.num_bands, ds.image_size, ds.image_size)
        hds = images_file.create_dataset(const.image_h5_name, hds_shape, dtype=ds.dtype)
        for i in range(num_images):
            random_idx = random.randrange(len(ds))
            image = ds[random_idx]['image']
            background = ds[random_idx]['background']
            hds[i, :, :, :] = image
            hds.flush()
        hds.attrs[const.background_h5_name] = background
        hds.flush()
