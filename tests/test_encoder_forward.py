import torch

from bliss.encoder import Encoder


class TestEncoder:
    def test_variational_mode(self, devices):
        """Tests forward function of source encoder.

        Arguments:
            devices: GPU device information.
        """
        device = devices.device

        batch_size = 2
        n_tiles_h = 3
        n_tiles_w = 5
        # TODO: we can no longer detect 4 source per tile; now we can only detect 1
        max_detections = 4
        tiles_to_crop = 2
        n_bands = 2
        tile_slen = 2
        background = (10.0, 20.0)

        encoder: Encoder = Encoder(
            # TODO: Let's use this directory's config.yml file
            # for these tests, in part so we can specify a nontrivial encoder architecture
            architecture=[[-1, 1, "Conv", [64, 5, 1]]],
            tiles_to_crop=tiles_to_crop,
            tile_slen=tile_slen,
            n_bands=n_bands,
        ).to(device)

        with torch.no_grad():
            encoder.eval()

            # simulate image padded tiles
            images = torch.randn(
                batch_size,
                n_bands,
                (n_tiles_h + 2 * tiles_to_crop) * tile_slen,
                (n_tiles_w - 2 * tiles_to_crop) * tile_slen,
                device=device,
            )

            background_tensor = torch.tensor(background, device=device)
            background_tensor = background_tensor.reshape(1, -1, 1, 1).expand(*images.shape)

            images *= background_tensor.sqrt()
            images += background_tensor
            batch = {"images": images, "background": background_tensor}

            var_params = encoder.encode_batch(batch)
            catalog = encoder.variational_mode(var_params)

            assert catalog["n_sources"].size() == torch.Size([batch_size * n_tiles_h * n_tiles_w])
            correct_locs_shape = torch.Size([batch_size * n_tiles_h * n_tiles_w, max_detections, 2])
            assert catalog["locs"].shape == correct_locs_shape
