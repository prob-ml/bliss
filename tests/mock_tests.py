import base64
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from bliss.surveys.sdss import SDSSDownloader


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


class MockSDSSDownloader(SDSSDownloader):
    def __init__(self, image_ids, download_dir):
        """Create a mock SDSSDownloader that copies local data instead of downloading.
        Inits SDSSDownloader for the first field in the list.

        Args:
            image_ids (int, int, int): list of (run, camcol, field) to download
            download_dir (str): Directory to download to
        """
        # copy data from normal dir instead of downloading
        for image_id in image_ids:
            run, camcol, field = image_id
            pf_file = f"photoField-{run:06d}-{camcol:d}.fits"
            file_dst = Path(f"{download_dir}/{run}/{camcol}/{pf_file}")
            file_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                f"data/sdss/{run}/{camcol}/{pf_file}",
                file_dst,
            )

            dst = Path(f"{download_dir}/{run}/{camcol}/{field}")
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                f"data/sdss/{run}/{camcol}/{field}",
                dst,
                dirs_exist_ok=True,
            )

        super().__init__(image_ids, download_dir)


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


def cached_output_from_predict_fn(cfg, filename="sdss_preds.pt"):
    test_data_path = Path("data/tests")

    # copy prediction file to temp directory so tests can find it
    Path(cfg.predict.plot.out_file_name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test_data_path / "predict.html", cfg.predict.plot.out_file_name)

    # return catalog and preds like predict_sdss
    with open(test_data_path / filename, "rb") as f:
        data = torch.load(f)
    return data["catalog"], data["images"], data["backgrounds"], None, data["preds"]


def mock_predict(cfg, *args, **kwargs):
    return cached_output_from_predict_fn(cfg, filename="sdss_preds.pt")


def mock_predict_bulk(cfg, *args, **kwargs):
    return cached_output_from_predict_fn(cfg, filename="sdss_preds_bulk.pt")
