import torch
from hydra.utils import instantiate

from bliss.catalog import RegionCatalog


class TestEncoder:
    def test_encode_multi_source_catalog(self, cfg, multi_source_dataloader, encoder):
        batch = next(iter(multi_source_dataloader))
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(cfg.predict.device)
        pred = encoder.encode_batch(batch)
        encoder.variational_mode(pred)

    def test_encode_with_psf(self, cfg, multiband_dataloader):
        batch = next(iter(multiband_dataloader))
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(cfg.predict.device)

        encoder_params = {
            "bands": [2],
            "input_transform_params": {"use_deconv_channel": True, "concat_psf_params": True},
        }
        encoder = instantiate(cfg.encoder, **encoder_params).to(cfg.predict.device)
        pred = encoder.encode_batch(batch)
        encoder.variational_mode(pred)

    def test_region_loss(self, cfg, encoder):
        d = {
            "n_sources": torch.zeros(1, 7, 7),
            "locs": torch.zeros(1, 7, 7, 1, 2),
            "source_type": torch.zeros((1, 7, 7, 1, 1)).bool(),
            "galaxy_params": torch.zeros((1, 7, 7, 1, 6)),
            "star_fluxes": torch.zeros((1, 7, 7, 1, 5)),
            "galaxy_fluxes": torch.zeros(1, 7, 7, 1, 5),
        }
        region_cat = RegionCatalog(interior_slen=3.6, overlap_slen=0.4, d=d).to(cfg.predict.device)

        images = torch.rand(1, 5, 16, 16, device=cfg.predict.device)
        background = torch.rand(1, 5, 16, 16, device=cfg.predict.device)
        batch = {"images": images, "background": background}

        encoder.tiles_to_crop = 0
        encoder.eval()
        with torch.no_grad():
            pred = encoder.encode_batch(batch)
            encoder._get_loss(pred, region_cat)  # pylint: disable=W0212  # noqa: WPS437

    def test_region_locs_in_tiles(self):
        d = {
            "n_sources": torch.ones(1, 5, 5),
            "locs": torch.ones(1, 5, 5, 1, 2) * 0.6,
        }
        cat = RegionCatalog(interior_slen=3.6, overlap_slen=0.4, d=d)

        interior_locs = cat.get_interior_locs_in_tile()
        locs_left, locs_right = cat.get_vertical_boundary_locs_in_tiles()
        locs_up, locs_down = cat.get_horizontal_boundary_locs_in_tiles()

        assert torch.allclose(
            interior_locs[0, 0, 0], torch.tensor([0.6 * 3.8 / 4.2, 0.6 * 3.8 / 4.2])
        )
        assert torch.allclose(
            interior_locs[0, 0, 4], torch.tensor([0.6 * 3.8 / 4.2, (0.6 * 3.8 + 0.4) / 4.2])
        )
        assert torch.allclose(
            interior_locs[0, 2, 2],
            torch.tensor([(0.6 * 3.6 + 0.4) / 4.4, ((0.6 * 3.6 + 0.4) / 4.4)]),
        )

        assert torch.allclose(
            locs_left[0, 0, 1], torch.tensor([0.6 * 3.8 / 4.2, (0.6 * 0.4 + 3.8) / 4.2])
        )
        assert torch.allclose(
            locs_right[0, 0, 1], torch.tensor([0.6 * 3.8 / 4.2, (0.6 * 0.4) / 4.4])
        )
        assert torch.allclose(
            locs_left[0, 4, 3], torch.tensor([(0.6 * 3.8 + 0.4) / 4.2, (0.6 * 0.4 + 4) / 4.4])
        )
        assert torch.allclose(
            locs_right[0, 4, 3], torch.tensor([(0.6 * 3.8 + 0.4) / 4.2, (0.6 * 0.4) / 4.2])
        )

        assert torch.allclose(
            locs_up[0, 1, 0], torch.tensor([(0.6 * 0.4 + 3.8) / 4.2, 0.6 * 3.8 / 4.2])
        )
        assert torch.allclose(locs_down[0, 1, 0], torch.tensor([0.6 * 0.4 / 4.4, 0.6 * 3.8 / 4.2]))
        assert torch.allclose(
            locs_up[0, 3, 2], torch.tensor([(0.6 * 0.4 + 4) / 4.4, (0.6 * 3.6 + 0.4) / 4.4])
        )
        assert torch.allclose(
            locs_down[0, 3, 2], torch.tensor([0.6 * 0.4 / 4.2, (0.6 * 3.6 + 0.4) / 4.4])
        )
