from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn


class ImageBackground(nn.Module):
    def __init__(
        self,
        image_items,
        bands: Tuple[int, ...],
    ):
        """Construct a background image from a set of images.

        Args:
            image_items: list of survey image items from which to simulate the background.
                Needs to contain a "background" key with a Tensor of background images.
            bands: bands to use for constructing the background, passed from Survey
        """

        super().__init__()

        backgrounds = []
        for image_item in image_items:
            backgrounds.append(image_item["background"][list(bands)])
        background = torch.from_numpy(np.stack(backgrounds, axis=0))

        self.register_buffer("background", background, persistent=False)
        self.height, self.width = self.background.shape[-2:]

    def sample(self, shape, image_id_indices) -> Tensor:
        assert isinstance(self.background, Tensor)
        batch_size, n_bands, hlen, wlen = shape
        assert self.background.shape[1] == n_bands
        assert image_id_indices.shape[0] == batch_size

        # select region to sample from (same for all images in batch for simplicity)
        h_diff, w_diff = self.height - hlen, self.width - wlen
        h = 0 if h_diff == 0 else np.random.randint(h_diff)
        w = 0 if w_diff == 0 else np.random.randint(w_diff)

        # sample region from specified background for each image in batch
        return self.background[image_id_indices, :, h : (h + hlen), w : (w + wlen)]
