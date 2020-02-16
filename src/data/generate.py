import numpy as np
import h5py
import random

from src.data import galaxy_datasets
from src import utils


def generate_images(dataset_name, outdir_name, outfile_name, num_images=1,
                    slen=40, num_bands=6, fixed_size=False, sky_factor=20, prop_name="prop.txt"):
    """

    :param dataset_name: The name of the dataset from :mod:`galaxy_datasets` that you want to use.
    :param num_images: number of images to save.
    :param outdir_name: The directory where you want to save the images in the `processed` directory.
    :return:
    """

    output_path = utils.data_path.joinpath(f"processed/{outdir_name}")
    output_path.mkdir(exist_ok=True)

    ds = galaxy_datasets.decide_dataset(dataset_name, slen, num_bands, fixed_size=fixed_size,
                                        sky_factor=sky_factor)

    # save the properties of the dataset used.
    prop_file_path = output_path.joinpath(prop_name)
    with open(prop_file_path, 'w') as prop_file:
        ds.print_props(prop_file)

    image_file_path = output_path.joinpath(f"{outfile_name}.hdf5")

    with h5py.File(image_file_path, "w") as images_file:
        for i in range(num_images):
            random_idx = random.randrange(len(ds))
            image = ds[random_idx]['image']
            background = ds[random_idx]['background']
            hds = images_file.create_dataset(utils.image_h5_name.format(i), image.shape, dtype=image.dtype)
            hds[:, :, :] = image
            hds.flush()
        hds = images_file.create_dataset('background', background.shape, dtype=background.dtype)
        hds[:, :, :] = background
        hds.flush()

