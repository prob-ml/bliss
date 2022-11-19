"""Scripts to produce BLISS estimates on astronomical images."""
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from tqdm import tqdm

from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources
from bliss.models.binary import BinaryEncoder
from bliss.models.detection_encoder import DetectionEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.models.galsim_encoder import GalsimEncoder
from bliss.models.lens_encoder import LensEncoder
from bliss.models.lensing_binary_encoder import LensingBinaryEncoder


class Encoder(nn.Module):
    """Encodes astronomical image into variational parameters.

    This module takes an astronomical image, or specifically padded tiles
    of an astronomical image, and returns either samples from the variational
    distribution of the latent catalog of objects represented by that image.

    Alternatively, this module can also return a sequential 'maximum-a-posteriori'
    (though this is not the true MAP since estimation is done sequentially rather than
    for the joint distribution or parameters).

    Attributes:
        See the __init__ function for a description of the attributes, which are
        the submodules for specific components of the catalog.
    """

    def __init__(
        self,
        detection_encoder: DetectionEncoder,
        binary_encoder: Optional[BinaryEncoder] = None,
        galaxy_encoder: Optional[Union[GalaxyEncoder, GalsimEncoder]] = None,
        lensing_binary_encoder: Optional[LensingBinaryEncoder] = None,
        lens_encoder: Optional[LensEncoder] = None,
        n_images_per_batch: Optional[int] = None,
        n_rows_per_batch: Optional[int] = None,
        map_n_source_weights: Optional[Tuple[float, ...]] = None,
    ):
        """Initializes Encoder.

        This module requires at least the `detection_encoder`. Other
        modules can be incorporated to add more information about the catalog,
        specifically whether an object is a galaxy or star (`binary_encoder`), or
        the latent parameter describing the shape of the galaxy `galaxy_encoder`.

        Args:
            detection_encoder: Module that takes padded tiles and returns the number
                of sources and locations per-tile.
            binary_encoder: Module that takes padded tiles and locations and
                returns a classification between stars and galaxies. Defaults to None.
            lensing_binary_encoder: Module that takes padded tiles and locations and
                returns a classification between lensed and unlensed galaxies. Defaults to None.
            galaxy_encoder: Module that takes padded tiles and locations and returns the variational
                distribution of the latent variable determining the galaxy shape. Defaults to None.
            lens_encoder: Module that takes padded tiles and locations and returns the variational
                distribution of the latent variables determining the lens shape. Defaults to None.
            n_images_per_batch: How many images can be processed at a time on the GPU?
                If not specified, defaults to an amount known to fit on my GPU.
            n_rows_per_batch: How many vertical padded tiles can be processed at a time on the GPU?
                If not specified, defaults to an amount known to fit on my GPU.
            map_n_source_weights: Optional. See DetectionEncoder. If specified, weights the argmax
                in MAP estimation of locations. Useful for raising/lowering the threshold for
                turning sources on/off.
        """
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.detection_encoder = detection_encoder
        self.binary_encoder = binary_encoder
        self.lensing_binary_encoder = lensing_binary_encoder
        self.galaxy_encoder = galaxy_encoder
        self.lens_encoder = lens_encoder

        if self.lens_encoder is not None:
            # need to have lensing classifier and source galaxy encoder if lensing is desired
            assert self.lensing_binary_encoder is not None

        if map_n_source_weights is None:
            map_n_source_weights_tnsr = torch.ones(self.detection_encoder.max_detections + 1)
        else:
            map_n_source_weights_tnsr = torch.tensor(map_n_source_weights)

        self.n_images_per_batch = n_images_per_batch if n_images_per_batch is not None else 10
        self.n_rows_per_batch = n_rows_per_batch if n_rows_per_batch is not None else 15
        self.register_buffer("map_n_source_weights", map_n_source_weights_tnsr, persistent=False)

    def forward(self, x):
        raise NotImplementedError("Unavailable. Use .variational_mode() or .sample() instead.")

    def variational_mode(self, image: Tensor, background: Tensor) -> TileCatalog:
        """Get maximum a posteriori of catalog from image padded tiles.

        Note that, strictly speaking, this is not the true MAP of the variational
        distribution of the catalog.

        Rather, we use sequential estimation; the MAP of the locations is first estimated,
        then plugged-in to the binary and galaxy encoders. Thus, the binary and galaxy
        encoders are conditioned on the location MAP. The true MAP would require optimizing
        over the entire catalog jointly, but this is not tractable.

        Args:
            image: An astronomical image,
                with shape `n * n_bands * h * w`.
            background: Background associated with image,
                with shape `n * n_bands * h * w`.

        Returns:
            A dictionary of the maximum a posteriori
            of the catalog in tiles. Specifically, this dictionary comprises:
                - The output of DetectionEncoder.variational_mode()
                - 'galaxy_bools', 'star_bools', and 'galaxy_probs' from BinaryEncoder.
                - 'galaxy_params' from GalaxyEncoder.
                - 'lens_params' from LensEncoder.
        """
        tile_map_dict = self.sample(image, background, None)
        n_tiles_h = (image.shape[2] - 2 * self.border_padding) // self.detection_encoder.tile_slen
        n_tiles_w = (image.shape[3] - 2 * self.border_padding) // self.detection_encoder.tile_slen
        return TileCatalog.from_flat_dict(
            self.detection_encoder.tile_slen,
            n_tiles_h,
            n_tiles_w,
            {k: v.squeeze(0) for k, v in tile_map_dict.items()},
        )

    def sample(
        self, image: Tensor, background: Tensor, n_samples: Optional[int]
    ) -> Dict[str, Tensor]:
        n_tiles_h = (image.shape[2] - 2 * self.border_padding) // self.detection_encoder.tile_slen
        ptile_loader = self.make_ptile_loader(image, background, n_tiles_h)
        tile_map_list: List[Dict[str, Tensor]] = []
        with torch.no_grad():
            for ptiles in tqdm(ptile_loader, desc="Encoding ptiles"):
                out_ptiles = self._encode_ptiles(ptiles, n_samples)
                tile_map_list.append(out_ptiles)
        return self.collate(tile_map_list)

    def make_ptile_loader(self, image: Tensor, background: Tensor, n_tiles_h: int):
        img_bg = torch.cat((image, background), dim=1).to(self.device)
        n_images = image.shape[0]
        for start_b in range(0, n_images, self.n_images_per_batch):
            for row in range(0, n_tiles_h, self.n_rows_per_batch):
                end_b = start_b + self.n_images_per_batch
                end_row = row + self.n_rows_per_batch
                start_h = row * self.detection_encoder.tile_slen
                end_h = end_row * self.detection_encoder.tile_slen + 2 * self.border_padding
                img_bg_cropped = img_bg[start_b:end_b, :, start_h:end_h, :]
                image_ptiles = get_images_in_tiles(
                    img_bg_cropped,
                    self.detection_encoder.tile_slen,
                    self.detection_encoder.ptile_slen,
                )
                yield image_ptiles.reshape(-1, *image_ptiles.shape[-3:])

    def _encode_ptiles(self, image_ptiles: Tensor, n_samples: Optional[int]):
        assert isinstance(self.map_n_source_weights, Tensor)
        deterministic = n_samples is None
        dist_params = self.detection_encoder.encode(image_ptiles)
        tile_samples = self.detection_encoder.sample(
            dist_params, n_samples, n_source_weights=self.map_n_source_weights
        )
        locs = tile_samples["locs"]
        n_sources = tile_samples["n_sources"]
        is_on_array = get_is_on_from_n_sources(n_sources, self.detection_encoder.max_detections)

        # add some variational distribution parameters to output
        n_source_log_probs = dist_params["n_source_log_probs"][:, 1:]
        tile_samples["n_source_log_probs"] = n_source_log_probs.unsqueeze(-1).unsqueeze(0)
        dist_params_n_src = self.detection_encoder.encode_for_n_sources(
            dist_params["per_source_params"], n_sources
        )
        tile_samples["log_flux_sd"] = dist_params_n_src["log_flux_sd"]

        if self.binary_encoder is not None:
            assert not self.binary_encoder.training
            galaxy_probs = self.binary_encoder.forward(image_ptiles, locs)
            galaxy_probs *= is_on_array.unsqueeze(-1)
            if deterministic:
                galaxy_bools = (galaxy_probs > 0.5).float() * is_on_array.unsqueeze(-1)
            else:
                galaxy_bools = (torch.rand_like(galaxy_probs) > 0.5).float()
                galaxy_bools *= is_on_array.unsqueeze(-1)

            tile_samples.update({"galaxy_bools": galaxy_bools, "galaxy_probs": galaxy_probs})

            if self.lensing_binary_encoder is not None:
                assert not self.lensing_binary_encoder.training
                lensed_galaxy_probs = self.lensing_binary_encoder.forward(image_ptiles, locs)
                lensed_galaxy_probs *= is_on_array.unsqueeze(-1)

                if deterministic:
                    lensed_galaxy_bools = (lensed_galaxy_probs > 0.5).float()
                    lensed_galaxy_bools *= is_on_array.unsqueeze(-1)
                else:
                    lensed_galaxy_bools = (torch.rand_like(lensed_galaxy_probs) > 0.5).float()
                    lensed_galaxy_bools *= is_on_array.unsqueeze(-1)

                # currently only support lensing where galaxy is present
                lensed_galaxy_bools *= galaxy_bools

                tile_samples["lensed_galaxy_bools"] = lensed_galaxy_bools
                tile_samples["lensed_galaxy_probs"] = lensed_galaxy_probs

        if self.galaxy_encoder is not None:
            galaxy_params = self.galaxy_encoder.sample(
                image_ptiles, locs, deterministic=deterministic
            )
            galaxy_params *= is_on_array.unsqueeze(-1) * galaxy_bools
            tile_samples.update({"galaxy_params": galaxy_params})

        if self.lens_encoder is not None:
            lens_params = self.lens_encoder.sample(image_ptiles, locs)
            lens_params *= is_on_array.unsqueeze(-1) * lensed_galaxy_bools
            tile_samples.update({"lens_params": lens_params})

        return tile_samples

    @staticmethod
    def collate(tile_map_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for k in tile_map_list[0]:
            out[k] = torch.cat([d[k] for d in tile_map_list], dim=1)
        return out

    def get_images_in_ptiles(self, images):
        """Run get_images_in_ptiles with correct tile_slen and ptile_slen."""
        return get_images_in_tiles(
            images, self.detection_encoder.tile_slen, self.detection_encoder.ptile_slen
        )

    @property
    def border_padding(self) -> int:
        return self.detection_encoder.border_padding

    @property
    def device(self):
        return self._dummy_param.device
