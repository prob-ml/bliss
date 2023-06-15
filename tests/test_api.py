import os

import pytest

from bliss.api import BlissClient


@pytest.fixture(scope="session")
def cwd(tmpdir_factory):
    return tmpdir_factory.mktemp("cwd")


@pytest.fixture(scope="class")
def bliss_client(cwd):
    return BlissClient(str(cwd))


@pytest.fixture(scope="class")
def cached_data_path(bliss_client):
    bliss_client.cached_data_path = bliss_client.cwd + "/dataset"
    bliss_client.generate(
        n_batches=3,
        batch_size=64,
        max_images_per_file=128,
        training={"trainer": {"accelerator": "cpu"}},
    )
    return bliss_client.cached_data_path


@pytest.fixture(scope="class")
def pretrained_weights_filename(bliss_client):
    assert os.path.exists(
        bliss_client.pretrained_weights_path
    ), f"pretrained_weights_path {bliss_client.pretrained_weights_path} not found"
    filename = "sdss_pretrained_fixture.pt"
    bliss_client.load_pretrained_weights_for_survey(
        survey="sdss",
        filename=filename,
    )
    return filename


@pytest.fixture(scope="class")
def weight_save_path(bliss_client, pretrained_weights_filename):
    weight_save_path = "tutorial_encoder/0_fixture.pt"
    bliss_client.train_on_cached_data(
        weight_save_path=weight_save_path,
        train_n_batches=2,
        batch_size=64,
        val_split_file_idxs=[1],
        pretrained_weights_filename=pretrained_weights_filename,
        training={"trainer": {"accelerator": "cpu"}},
    )
    return weight_save_path


@pytest.mark.usefixtures(
    "bliss_client", "cached_data_path", "pretrained_weights_filename", "weight_save_path"
)
class TestApi:
    def test_generate(self, bliss_client):
        bliss_client.generate(n_batches=3, batch_size=64, max_images_per_file=128)
        # alter default cached_data_path
        bliss_client.cached_data_path = bliss_client.cwd + "/dataset_ms0.02"
        bliss_client.generate(
            n_batches=3,
            batch_size=64,
            max_images_per_file=128,
            simulator={"prior": {"mean_sources": 0.02}},  # optional
            generate={"file_prefix": "dataset"},  # optional
            training={"trainer": {"accelerator": "cpu"}},
        )
        # check that cached datasets generated
        assert os.path.exists(
            bliss_client.cwd + "/dataset/dataset_0.pt"
        ), "{CWD}/dataset/dataset_0.pt not found"
        assert os.path.exists(
            bliss_client.cwd + "/dataset_ms0.02/dataset_0.pt"
        ), "{CWD}/dataset_ms0.02/dataset_0.pt not found"

    def test_get_dataset_file(self, bliss_client, cached_data_path):
        bliss_client.cached_data_path = cached_data_path
        dataset_ms0p02_0 = bliss_client.get_dataset_file(filename="dataset_0.pt")
        assert isinstance(dataset_ms0p02_0, list), "dataset_ms0p02_0 must be a list"
        bliss_client.cached_data_path = bliss_client.cwd + "/dataset"
        dataset0 = bliss_client.get_dataset_file(filename="dataset_0.pt")
        assert isinstance(dataset0, list), "dataset0 must be a list"

    def test_load_pretrained_weights(self, bliss_client):
        assert os.path.exists(
            bliss_client.pretrained_weights_path
        ), f"pretrained_weights_path {bliss_client.pretrained_weights_path} not found"
        bliss_client.load_pretrained_weights_for_survey(
            survey="sdss", filename="sdss_pretrained.pt"
        )

    def test_train_on_cached_data(self, bliss_client, pretrained_weights_filename):
        bliss_client.train_on_cached_data(
            weight_save_path="tutorial_encoder/0.pt",
            train_n_batches=2,
            batch_size=64,
            val_split_file_idxs=[1],
            pretrained_weights_filename=pretrained_weights_filename,
            training={"trainer": {"accelerator": "cpu"}},
        )

    def test_predict_sdss_default_rcf(self, bliss_client, weight_save_path):
        bliss_client.predict_sdss(
            data_path="data/sdss",
            weight_save_path=weight_save_path,
            training={"trainer": {"accelerator": "cpu"}},
            predict={"device": "cpu"},
        )
        bliss_client.plot_predictions_in_notebook()

    def test_predict_sdss_custom_rcf(self, bliss_client, weight_save_path):
        bliss_client.predict_sdss(
            data_path="data/sdss",
            weight_save_path=weight_save_path,
            predict={"dataset": {"run": 1011, "camcol": 3, "fields": [44]}, "device": "cpu"},
            training={"trainer": {"accelerator": "cpu"}},
        )
