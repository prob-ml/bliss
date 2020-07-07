import random
import inspect
import sys
import h5py

from torch.utils.data import Dataset


def save_images(
    dataset, file_path, n_images=1,
):
    """Generate images from dataset cls and save num_images into h5py file.
    """

    with h5py.File(file_path, "w") as images_file:
        hds_shape = (n_images, dataset.n_bands, dataset.slen, dataset.slen)
        hds = images_file.create_dataset("images", hds_shape, dtype=dataset.dtype)
        for i in range(n_images):
            random_idx = random.randrange(len(dataset))
            image = dataset[random_idx]["image"]
            hds[i, :, :, :] = image
            hds.flush()
        hds.attrs["background"] = dataset.background
        hds.flush()


class H5Catalog(Dataset):
    def __init__(self, h5_file="images.hdf5", slen=51, n_bands=1):
        """ A dataset created from single galaxy images in a h5py file.
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
        """Number of images saved in the file.
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

    @staticmethod
    def add_args(parser):
        parser.add_argument("--h5-file", type=str, default=None, help="file path")

    @classmethod
    def from_args(cls, args):
        assert args.h5_file, "Specify h5_file if using this dataset."

        args_dict = vars(args)
        parameters = inspect.signature(cls).parameters
        args_dict = {param: args_dict[param] for param in parameters}
        return cls(**args_dict)
