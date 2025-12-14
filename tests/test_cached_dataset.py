import torch

from bliss.cached_dataset import (
    ChunkingDataset,
    ChunkingSampler,
    FluxFilterTransform,
    FullCatalogToTileTransform,
    OneBandTransform,
)
from bliss.catalog import TileCatalog


def make_sample(n_bands=3, img_size=16):
    return {
        "images": torch.randn(n_bands, img_size, img_size),
        "background": torch.zeros(n_bands, img_size, img_size),
        "psf_params": torch.zeros(n_bands, 6),
        "full_catalog": {
            "n_sources": torch.tensor(2),
            "plocs": torch.tensor([[4.0, 4.0], [12.0, 12.0]]),
            "source_type": torch.ones(2, 1).bool(),
            "fluxes": torch.tensor([[100.0, 50.0, 25.0], [200.0, 100.0, 50.0]]),
        },
    }


class TestChunkingDataset:
    def test_length_from_filename(self, tmp_path):
        data = [make_sample() for _ in range(5)]
        path = tmp_path / "data_size_5.pt"
        torch.save(data, path)
        dataset = ChunkingDataset([str(path)], transform=lambda x: x)
        assert len(dataset) == 5

    def test_getitem(self, tmp_path):
        data = [make_sample() for _ in range(3)]
        path = tmp_path / "data_size_3.pt"
        torch.save(data, path)
        dataset = ChunkingDataset([str(path)], transform=lambda x: x)
        item = dataset[0]
        assert "images" in item
        assert item["images"].shape == (3, 16, 16)

    def test_multiple_files(self, tmp_path):
        data1 = [make_sample() for _ in range(2)]
        data2 = [make_sample() for _ in range(3)]
        path1 = tmp_path / "data1_size_2.pt"
        path2 = tmp_path / "data2_size_3.pt"
        torch.save(data1, path1)
        torch.save(data2, path2)
        dataset = ChunkingDataset([str(path1), str(path2)], transform=lambda x: x)
        assert len(dataset) == 5
        assert "images" in dataset[0]
        assert "images" in dataset[2]


class TestChunkingSampler:
    def test_sampler_length(self, tmp_path):
        data = [make_sample() for _ in range(4)]
        path = tmp_path / "data_size_4.pt"
        torch.save(data, path)
        dataset = ChunkingDataset([str(path)], transform=lambda x: x)
        sampler = ChunkingSampler(dataset)
        assert len(sampler) == 4

    def test_sampler_iteration(self, tmp_path):
        data = [make_sample() for _ in range(4)]
        path = tmp_path / "data_size_4.pt"
        torch.save(data, path)
        dataset = ChunkingDataset([str(path)], shuffle=False, transform=lambda x: x)
        sampler = ChunkingSampler(dataset)
        indices = list(sampler)
        assert indices == [0, 1, 2, 3]


class TestFullCatalogToTileTransform:
    def test_produces_tile_catalog(self):
        transform = FullCatalogToTileTransform(tile_slen=4, max_sources=2)
        sample = make_sample(img_size=16)
        result = transform(sample)
        assert "tile_catalog" in result
        assert "full_catalog" not in result
        cat = TileCatalog.from_dict(result["tile_catalog"])
        assert cat.n_tiles_h == 4
        assert cat.n_tiles_w == 4


class TestOneBandTransform:
    def test_extracts_band(self):
        transform = OneBandTransform(band_idx=1)
        sample = make_sample(n_bands=5)
        sample["tile_catalog"] = {
            "n_sources": torch.tensor([[1, 0], [0, 1]]),
            "locs": torch.rand(2, 2, 1, 2),
            "source_type": torch.ones(2, 2, 1, 1).bool(),
            "fluxes": torch.rand(2, 2, 1, 5),
        }
        result = transform(sample)
        assert result["images"].shape[0] == 1
        assert result["psf_params"].shape[0] == 1
        assert result["tile_catalog"]["fluxes"].shape[-1] == 1


class TestFluxFilterTransform:
    def test_filters_low_flux(self):
        transform = FluxFilterTransform(reference_band=0, min_flux=150.0)
        sample = make_sample()
        tile_transform = FullCatalogToTileTransform(tile_slen=4, max_sources=2)
        sample = tile_transform(sample)
        result = transform(sample)
        cat = TileCatalog.from_dict(result["tile_catalog"])
        total_sources = cat["n_sources"].sum().item()
        assert total_sources == 1
