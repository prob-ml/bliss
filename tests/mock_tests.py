import base64
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from bliss.catalog import TileCatalog


class MockGetResponse:
    def __init__(self):
        self.content = base64.b64encode(b"test\n :\n 1")

    def json(self):
        return {"sha": "sha", "content": self.content, "encoding": "base64"}


class MockPostResponse:
    def json(self):
        return {"objects": [{"actions": {"download": {"href": "test"}}}]}


class MockSimulator(IterableDataset):
    def __iter__(self):
        yield {}

    def train_dataloader(self):
        return DataLoader(self)


class MockTrainer:
    def __init__(self, *args, **kwargs):
        self.logger = kwargs.get("logger", None)

    def fit(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass


class MockCallback:
    def __init__(self, *args, **kwargs):
        self.best_model_path = "data/tests/test_checkpoint.pt"


def mock_get(*args, **kwargs):
    return MockGetResponse()


def mock_post(*args, **kwargs):
    return MockPostResponse()


def mock_simulator(*args, **kwargs):
    return MockSimulator()


def mock_itemize_data(*args, **kwargs):
    return []


def mock_trainer(*args, **kwargs):
    return MockTrainer(*args, **kwargs)


def mock_checkpoint_callback(*args, **kwargs):
    return MockCallback()


def mock_predict_sdss(cfg, *args, **kwargs):
    test_data_path = "data/tests"

    # copy prediction file to temp directory so tests can find it
    Path(cfg.predict.plot.out_file_name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test_data_path + "/predict.html", cfg.predict.plot.out_file_name)

    # return catalog and preds like predict_sdss
    with open(test_data_path + "/sdss_preds.pt", "rb") as f:
        data = torch.load(f)
    tile_cat = TileCatalog(cfg.simulator.prior.tile_slen, data["catalog"])
    return tile_cat, data["image"], data["background"], None, data["pred"]
