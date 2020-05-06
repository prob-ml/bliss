import torch
from torch.utils.data import Dataset
from ..models import galaxy_net


class DecoderSamples(Dataset):
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
        assert torch.cuda.is_available(), "Need GPU."

        self.dec = galaxy_net.CenteredGalaxyDecoder(slen, latent_dim, num_bands).cuda()
        self.dec.load_state_dict(torch.load(decoder_file.as_posix()))
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

    def sample(self, n_samples):
        # returns = (z, images) where z.shape = (n_samples, latent_dim) and images.shape =
        # (n_samples, n_bands, slen, slen)

        return self.dec.get_sample(n_samples, return_latent=True)
