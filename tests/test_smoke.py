import numpy as np
import torch
from astropy.wcs import WCS

from bliss import cached_dataset, catalog, data_augmentation, global_env, main, make_range
from bliss.align import align
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
from bliss.simulator.prior import CatalogPrior


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


class TestMakeRange:
    def test_basic_range(self):
        result = make_range(0, 5, 1)
        assert list(result) == [0, 1, 2, 3, 4]

    def test_range_with_exclusions(self):
        result = make_range(0, 5, 1, 2, 4)
        assert list(result) == [0, 1, 3]

    def test_range_with_nonexistent_exclusion(self):
        result = make_range(0, 5, 1, 10)
        assert list(result) == [0, 1, 2, 3, 4]


class TestAlign:
    def test_align_identity(self):
        h, w = 16, 16
        img = np.random.randn(2, 3, h, w).astype(np.float32)

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [w / 2, h / 2]
        wcs.wcs.cdelt = [1.0, 1.0]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        wcs_list = [[wcs for _ in range(3)] for _ in range(2)]
        result = align(img, wcs_list, ref_band=1, ref_depth=0)

        assert result.shape == img.shape  # pylint: disable=comparison-with-callable
        assert str(result.dtype) == "float32"


class TestCatalogPrior:
    def test_sample(self):
        prior_obj = CatalogPrior(
            survey_bands=["g", "r", "i"],
            n_tiles_h=2,
            n_tiles_w=2,
            batch_size=2,
            min_sources=0,
            max_sources=3,
            mean_sources=1.5,
            prob_galaxy=0.5,
            star_flux={"exponent": 1.5, "truncation": 100.0, "loc": 0.0, "scale": 100.0},
            galaxy_flux={"exponent": 1.5, "truncation": 100.0, "loc": 0.0, "scale": 100.0},
            galaxy_a_concentration=2.0,
            galaxy_a_loc=0.5,
            galaxy_a_scale=0.2,
            galaxy_a_bd_ratio=0.5,
            star_color_model_path="tests/data/sdss/color_models/star_gmm_nmgy.pkl",
            gal_color_model_path="tests/data/sdss/color_models/gal_gmm_nmgy.pkl",
            reference_band=1,
        )
        tile_catalog = prior_obj.sample()
        assert tile_catalog.batch_size == 2
        assert tile_catalog.n_tiles_h == 2
        assert tile_catalog.n_tiles_w == 2
        assert "locs" in tile_catalog
        assert "n_sources" in tile_catalog
        assert "source_type" in tile_catalog
        assert "fluxes" in tile_catalog
        assert "galaxy_disk_frac" in tile_catalog
