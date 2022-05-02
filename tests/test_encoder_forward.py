import torch
from einops import rearrange

from bliss.catalog import get_images_in_tiles
from bliss.models.location_encoder import LocationEncoder, LogBackgroundTransform


class TestSourceEncoder:
    def test_variational_mode(self, devices):
        """Tests forward function of source encoder.

        Arguments:
            devices: GPU device information.
        """
        device = devices.device

        batch_size = 2
        n_tiles_h = 3
        n_tiles_w = 5
        max_detections = 4
        ptile_slen = 10
        n_bands = 2
        tile_slen = 2
        background = (10.0, 20.0)

        # get encoder
        star_encoder: LocationEncoder = LocationEncoder(
            LogBackgroundTransform(),
            channel=8,
            dropout=0,
            spatial_dropout=0,
            hidden=64,
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
            n_bands=n_bands,
            mean_detections=0.48,
            max_detections=max_detections,
        ).to(device)

        with torch.no_grad():
            star_encoder.eval()

            # simulate image padded tiles
            images = torch.randn(
                batch_size,
                n_bands,
                ptile_slen + (n_tiles_h - 1) * tile_slen,
                ptile_slen + (n_tiles_w - 1) * tile_slen,
                device=device,
            )

            background_tensor = torch.tensor(background, device=device)
            background_tensor = background_tensor.reshape(1, -1, 1, 1).expand(*images.shape)

            images *= background_tensor.sqrt()
            images += background_tensor
            image_ptiles = get_images_in_tiles(
                torch.cat((images, background_tensor), dim=1),
                star_encoder.tile_slen,
                star_encoder.ptile_slen,
            )
            image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
            var_params = star_encoder.encode(image_ptiles)
            catalog = star_encoder.variational_mode(var_params)

            assert catalog["n_sources"].size() == torch.Size([batch_size * n_tiles_h * n_tiles_w])
            correct_locs_shape = torch.Size([batch_size * n_tiles_h * n_tiles_w, max_detections, 2])
            assert catalog["locs"].shape == correct_locs_shape

    def test_sample(self, devices):
        device = devices.device

        max_detections = 4
        ptile_slen = 10
        n_bands = 2
        tile_slen = 2
        n_samples = 5
        background = (10.0, 20.0)
        background_tensor = torch.tensor(background).view(1, -1, 1, 1).to(device)

        images = (
            torch.randn(1, n_bands, 4 * ptile_slen, 4 * ptile_slen).to(device)
            * background_tensor.sqrt()
            + background_tensor
        )
        background_tensor = background_tensor.expand(*images.shape)

        star_encoder: LocationEncoder = LocationEncoder(
            LogBackgroundTransform(),
            channel=8,
            dropout=0,
            spatial_dropout=0,
            hidden=64,
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
            n_bands=n_bands,
            mean_detections=0.48,
            max_detections=max_detections,
        ).to(device)

        image_ptiles = get_images_in_tiles(
            torch.cat((images, background_tensor), dim=1),
            star_encoder.tile_slen,
            star_encoder.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        var_params = star_encoder.encode(image_ptiles)
        star_encoder.sample(var_params, n_samples)
