import torch

from bliss import cached_dataset, catalog, data_augmentation, global_env, main
from bliss.catalog import FullCatalog, TileCatalog
from bliss.encoder import (
    convnet_layers,
    convnets,
    encoder,
    image_normalizer,
    metrics,
    variational_dist,
)
from bliss.encoder.convnets import FeaturesNet
from bliss.encoder.image_normalizer import (
    AsinhQuantileNormalizer,
    ClaheNormalizer,
    NullNormalizer,
    PsfAsImage,
)
from bliss.simulator import decoder, prior, psf


class TestImports:
    def test_import_all_modules(self):
        assert cached_dataset is not None
        assert catalog is not None
        assert data_augmentation is not None
        assert convnet_layers is not None
        assert convnets is not None
        assert encoder is not None
        assert image_normalizer is not None
        assert metrics is not None
        assert variational_dist is not None
        assert global_env is not None
        assert main is not None
        assert decoder is not None
        assert prior is not None
        assert psf is not None


class TestCatalogInstantiation:
    def test_tile_catalog_minimal(self):
        d = {
            "n_sources": torch.tensor([[[1, 0], [0, 1]]]),
            "locs": torch.rand(1, 2, 2, 1, 2),
            "source_type": torch.ones(1, 2, 2, 1, 1).bool(),
            "fluxes": torch.rand(1, 2, 2, 1, 3),
        }
        cat = TileCatalog(d)
        assert cat.batch_size == 1
        assert cat.n_tiles_h == 2
        assert cat.n_tiles_w == 2
        assert cat.max_sources == 1

    def test_full_catalog_minimal(self):
        d = {
            "n_sources": torch.tensor([2]),
            "plocs": torch.tensor([[[8.0, 8.0], [4.0, 4.0]]]),
            "source_type": torch.ones(1, 2, 1).bool(),
            "fluxes": torch.rand(1, 2, 3),
        }
        cat = FullCatalog(16, 16, d)
        assert cat.batch_size == 1
        assert cat.max_sources == 2
        assert cat.height == 16
        assert cat.width == 16


class TestNormalizers:
    def test_psf_as_image(self):
        normalizer = PsfAsImage(num_psf_params=6)
        batch = {
            "images": torch.randn(1, 3, 16, 16),
            "psf_params": torch.randn(1, 3, 6),
        }
        output = normalizer.get_input_tensor(batch)
        assert output.shape == (1, 3, 6, 16, 16)

    def test_clahe_normalizer(self):
        normalizer = ClaheNormalizer(min_stdev=0.001)
        batch = {"images": torch.randn(1, 3, 16, 16)}
        output = normalizer.get_input_tensor(batch)
        assert output.shape == (1, 3, 1, 16, 16)

    def test_asinh_quantile_normalizer(self):
        normalizer = AsinhQuantileNormalizer(q=[0.25, 0.5, 0.75])
        batch = {"images": torch.randn(1, 3, 16, 16)}
        output = normalizer.get_input_tensor(batch)
        assert output.shape == (1, 3, 3, 16, 16)

    def test_null_normalizer(self):
        normalizer = NullNormalizer()
        batch = {"images": torch.randn(1, 3, 16, 16)}
        output = normalizer.get_input_tensor(batch)
        assert output.shape == (1, 3, 1, 16, 16)


class TestFeaturesNet:
    def test_forward_pass(self):
        net = FeaturesNet(n_bands=3, ch_per_band=2, num_features=256, double_downsample=False)
        x = torch.randn(1, 3, 2, 64, 64)
        output = net(x)
        assert output.shape[0] == 1
        assert output.shape[1] == 256

    def test_forward_pass_double_downsample(self):
        net = FeaturesNet(n_bands=3, ch_per_band=2, num_features=256, double_downsample=True)
        x = torch.randn(1, 3, 2, 64, 64)
        output_dd = net(x)
        net_no_dd = FeaturesNet(n_bands=3, ch_per_band=2, num_features=256, double_downsample=False)
        output_no_dd = net_no_dd(x)
        assert output_dd.shape[2] < output_no_dd.shape[2]
