import logging
from os import environ, getenv
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, TypeAlias

import hydra
import torch
from astropy.table import Table
from omegaconf import OmegaConf

from bliss.catalog import FullCatalog
from bliss.generate import generate as _generate
from bliss.predict import predict as _predict
from bliss.surveys.sdss import SDSSDownloader
from bliss.train import train as _train
from bliss.utils.download_utils import download_git_lfs_file
from case_studies.api.igs import base_config

SurveyType: TypeAlias = Literal["decals", "hst", "dc2", "sdss"]


class BlissClient:
    """Client for interacting with the BLISS API."""

    def __init__(self, cwd: str):
        self.base_cfg = base_config()
        self._cwd = cwd
        self.base_cfg.paths.root = self.cwd

    def generate(
        self,
        n_batches: int,
        batch_size: int,
        max_images_per_file: int,
        **kwargs,
    ) -> None:
        """Generate and cache simulated images to disk.

        Args:
            n_batches (int): Number of batches to simulate.
            batch_size (int): Number of images per batch.
            max_images_per_file (int): Number of images per cached file.
            **kwargs: Keyword arguments to override default configuration values.
        """
        cfg = OmegaConf.create(self.base_cfg)
        # apply overrides
        cfg.generate.n_batches = n_batches
        cfg.generate.batch_size = batch_size
        cfg.generate.max_images_per_file = max_images_per_file
        cfg.generate.cached_data_path = self.cached_data_path
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        _generate(cfg.generate)

    def load_pretrained_weights_for_survey(
        self, survey: SurveyType, filename: str, request_headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Load pretrained weights for a survey.

        Args:
            survey (SurveyType): Survey to load pretrained weights for.
            filename (str): Name of pretrained weights file downloaded.
            request_headers (Optional[Dict[str, str]], optional): Request headers for downloading
                pretrained weights. Required if calling this function to load pretrained weights
                frequently (> 60 requests/hour, as of 2022-11-28). Defaults to None.
        """
        weights = download_git_lfs_file(
            "https://api.github.com/repos/prob-ml/bliss/contents/"
            f"data/pretrained_models/{survey}.pt",
            request_headers,
        )
        Path(self.base_cfg.paths.pretrained_models).mkdir(parents=True, exist_ok=True)
        with open(self.base_cfg.paths.pretrained_models + f"/{filename}", "wb+") as f:
            f.write(weights)

    def train(self, weight_save_path, **kwargs) -> None:
        """Train model on simulated images.

        Args:
            weight_save_path (str): Path to directory after cwd where trained model
                weights will be stored.
            **kwargs: Keyword arguments to override default configuration values.
        """
        cfg = OmegaConf.create(self.base_cfg)
        # apply overrides
        cfg.train.weight_save_path = cfg.paths.output + f"/{weight_save_path}"
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        _train(cfg.train)

    def train_on_cached_data(
        self,
        weight_save_path,
        splits,
        batch_size,
        pretrained_weights_filename: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Train on cached data.

        Args:
            weight_save_path (str): Path to directory after cwd where trained model
                weights will be stored.
            splits (str): train/val/test splits as percent ranges (e.g. "0:80/80:90/90:100")
            batch_size (int): Number of images per batch.
            pretrained_weights_filename (str): Name of pretrained weights file to load.
            **kwargs: Keyword arguments to override default configuration values.
        """
        cfg = OmegaConf.create(self.base_cfg)
        # apply overrides
        cfg.train.weight_save_path = cfg.paths.output + f"/{weight_save_path}"
        cfg.cached_simulator.splits = splits
        cfg.cached_simulator.batch_size = batch_size
        if pretrained_weights_filename is not None:
            cfg.train.pretrained_weights = (
                cfg.paths.pretrained_models + f"/{pretrained_weights_filename}"
            )
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)
        cfg.train.data_source = cfg.cached_simulator

        _train(cfg.train)

    def load_survey(self, survey: SurveyType, run, camcol, field, download_dir: str):
        SDSSDownloader([(run, camcol, field)], self.cwd + f"/{download_dir}").download_all()
        # assert files downloaded at download_dir

    def predict(
        self,
        survey: SurveyType,
        weight_save_path: str,
        **kwargs,
    ) -> Tuple[FullCatalog, Table, Dict[Any, Table]]:
        """Predict on `survey` images.

        Note that by default, one tile (4 pixels) is cropped from the edges of the image before
        making predictions, so the predicted locations will be systematically offset compared to
        the original image.

        Args:
            survey (SurveyType): Survey of image to predict on.
            weight_save_path (str): Path to directory after cwd where trained model
                weights are stored.
            **kwargs: Keyword arguments to override default configuration values.

        Returns:
            Tuple[FullCatalog, Table, Dict[Any, Table]]: Tuple of estimated catalog, estimated
                catalog as an astropy table, and probabilistic predictions catalogs as astropy
                tables
        """
        cfg = OmegaConf.create(self.base_cfg)
        # apply overrides
        cfg.predict.weight_save_path = cfg.paths.output + f"/{weight_save_path}"
        cfg.predict.dataset = "${surveys." + survey + "}"
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        return _predict(cfg.predict)

    def plot_predictions_in_notebook(self):
        """Plot predictions in notebook."""
        from IPython.core.display import HTML  # pylint: disable=import-outside-toplevel
        from IPython.display import display  # pylint: disable=import-outside-toplevel

        with open(self.base_cfg.predict.plot.out_file_name, "r", encoding="utf-8") as f:
            html_str = f.read()
            display(HTML(html_str))

    def get_dataset_file(self, filename: str):
        with open(self.cached_data_path + "/" + filename, "rb") as f:
            return torch.load(f)

    # Getters and setters
    @property
    def cwd(self) -> str:
        """Get current working directory.

        Returns:
            str: Current working directory.
        """
        return self._cwd

    @cwd.setter
    def cwd(self, cwd: str) -> None:
        """Set current working directory.

        Args:
            cwd (str): Current working directory.
        """
        self._cwd = cwd
        self.base_cfg.paths.root = self.cwd

    @property
    def cached_data_path(self) -> str:
        """Get path to cached data.

        Returns:
            str: Path to cached data.
        """
        return self.base_cfg.generate.cached_data_path

    @cached_data_path.setter
    def cached_data_path(self, cached_data_path: str) -> None:
        """Set path to cached data.

        Args:
            cached_data_path (str): Path to cached data.
        """
        self.base_cfg.generate.cached_data_path = cached_data_path


# pragma: no cover
# ============================== CLI ==============================


# config_path should be overriden when running `bliss` poetry executable
# e.g., `bliss -cp case_studies/summer_template -cn config`
@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def main(cfg):
    """Main entry point(s) for BLISS."""
    if not getenv("BLISS_HOME"):
        project_path = Path(__file__).resolve()
        bliss_home = project_path.parents[1]
        environ["BLISS_HOME"] = bliss_home.as_posix()

        logger = logging.getLogger(__name__)
        logger.warning(
            "WARNING: BLISS_HOME not set, setting to project root %s\n",  # noqa: WPS323
            environ["BLISS_HOME"],
        )

    bliss_client = BlissClient(cwd=cfg.paths.root)
    if cfg.mode == "generate":
        bliss_client.generate(
            n_batches=cfg.generate.n_batches,
            batch_size=cfg.generate.batch_size,
            max_images_per_file=cfg.generate.max_images_per_file,
            **cfg,
        )
    elif cfg.mode == "train":
        bliss_client.train(weight_save_path=cfg.train.weight_save_path, **cfg)
    elif cfg.mode == "predict":
        # survey="sdss" is hack since OmegaConf.update overwrites predict.dataset
        bliss_client.predict(survey="sdss", weight_save_path=cfg.predict.weight_save_path, **cfg)
    else:
        raise KeyError


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

# pragma: no cover
