import base64
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import requests
import torch
from astropy import units as u
from astropy.table import Table
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor

from bliss.catalog import FullCatalog
from bliss.conf.igs import base_config
from bliss.generate import generate as _generate
from bliss.predict import predict_sdss as _predict_sdss
from bliss.surveys import sdss_download
from bliss.train import train as _train

SurveyType = Literal["decals", "hst", "lsst", "sdss"]


class BlissClient:
    def __init__(self, cwd: str):
        self._cwd = cwd
        # cached_data_path (str): Path to directory where cached data will be stored.
        self._cached_data_path = self.cwd + "/dataset"
        Path(self.cached_data_path).mkdir(parents=True, exist_ok=True)
        # pretrained_weights_path (str): Path to directory to store pretrained weights.
        self._pretrained_weights_path = self.cwd + "/pretrained_weights"
        Path(self.pretrained_weights_path).mkdir(parents=True, exist_ok=True)
        # output_path (str): Path to directory to store trained model weights.
        self._output_path = self.cwd + "/output"
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

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
            kwargs: Keyword arguments to override default configuration values.
        """  # noqa: RST210
        cfg = base_config()
        # apply overrides
        cfg.generate.n_batches = n_batches
        cfg.generate.batch_size = batch_size
        cfg.generate.max_images_per_file = max_images_per_file
        cfg.generate.cached_data_path = self.cached_data_path
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        _generate(cfg)

    def load_pretrained_weights_for_survey(self, survey: SurveyType, filename: str) -> None:
        """Load pretrained weights for a survey.

        Args:
            survey (SurveyType): Survey to load pretrained weights for.
            filename (str): Name of pretrained weights file to load.
        """
        weights = _download_git_lfs_file(
            "https://api.github.com/repos/prob-ml/bliss/contents/"
            f"data/pretrained_models/{survey}.pt"
        )
        with open(self.pretrained_weights_path + f"/{filename}", "wb+") as f:
            f.write(weights)

    def train(self, weight_save_path, **kwargs) -> None:
        """Train model on simulated images.

        Args:
            weight_save_path (str): Path to directory after cwd where trained model
                weights will be stored.
            kwargs: Keyword arguments to override default configuration values.
        """  # noqa: RST210
        cfg = base_config()
        # apply overrides
        cfg.training.weight_save_path = self.output_path + f"/{weight_save_path}"
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        _train(cfg)

    def train_on_cached_data(
        self,
        weight_save_path,
        train_n_batches,
        batch_size,
        val_split_file_idxs,
        pretrained_weights_filename: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Train on cached data.

        Args:
            weight_save_path (str): Path to directory after cwd where trained model
                weights will be stored.
            train_n_batches (int): Number of batches to train on.
            batch_size (int): Number of images per batch.
            val_split_file_idxs (List[int]): List of file indices to use for validation.
            pretrained_weights_filename (str): Name of pretrained weights file to load.
            kwargs: Keyword arguments to override default configuration values.
        """  # noqa: RST210
        cfg = base_config()
        cfg.training.use_cached_simulator = True
        # apply overrides
        cfg.training.weight_save_path = self.output_path + f"/{weight_save_path}"
        cfg.cached_simulator.cached_data_path = self.cached_data_path
        cfg.cached_simulator.train_n_batches = train_n_batches
        cfg.cached_simulator.batch_size = batch_size
        cfg.cached_simulator.val_split_file_idxs = val_split_file_idxs
        if pretrained_weights_filename is not None:
            cfg.training.pretrained_weights = (
                self.pretrained_weights_path + f"/{pretrained_weights_filename}"
            )
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        _train(cfg)

    def load_survey(self, survey: SurveyType, run, camcol, field, download_dir: str):
        sdss_download.download_all(run, camcol, field, self.cwd + f"/{download_dir}")
        # assert files downloaded at download_dir

    def predict_sdss(
        self,
        data_path: str,
        weight_save_path: str,
        **kwargs,
    ) -> Tuple[FullCatalog, Table, Table, Table]:
        """Predict on SDSS images.

        Args:
            data_path (str): Path to directory after cwd where SDSS images are stored.
            weight_save_path (str): Path to directory after cwd where trained model
                weights are stored.
            kwargs: Keyword arguments to override default configuration values.

        Returns:
            Tuple[FullCatalog, Table, Table]: Tuple of predicted catalog, astropy.table
            of predicted catalog, and astropy.table of predicted galaxy_params.
        """
        cfg = base_config()
        # apply overrides
        cfg.predict.dataset.sdss_dir = self.cwd + f"/{data_path}"
        cfg.predict.weight_save_path = self.output_path + f"/{weight_save_path}"
        cfg.paths.sdss = cfg.predict.dataset.sdss_dir
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        # download survey images if not already downloaded
        run, camcol, field = (
            cfg.predict.dataset.run,
            cfg.predict.dataset.camcol,
            cfg.predict.dataset.fields[0],
        )
        bands = ["u", "g", "r", "i", "z"]
        for band in bands:
            sdss_data_file_path = (
                Path(cfg.predict.dataset.sdss_dir)
                / f"{run}"
                / f"{camcol}"
                / f"{field}"
                / f"frame-{band}-{'{:06d}'.format(run)}-{camcol}-{'{:04d}'.format(field)}.fits"
            )
            if not sdss_data_file_path.exists():
                self.load_survey("sdss", run, camcol, field, download_dir=data_path)
                break

        est_cat, _, _, _, pred = _predict_sdss(cfg)
        est_cat_table, galaxy_params_table = fullcat_to_astropy_table(est_cat)
        pred_table = pred_to_astropy_table(pred)
        return est_cat, est_cat_table, galaxy_params_table, pred_table

    def plot_predictions_in_notebook(self):
        """Plot predictions in notebook."""
        from IPython.core.display import HTML  # pylint: disable=import-outside-toplevel
        from IPython.display import display  # pylint: disable=import-outside-toplevel

        with open("./predict.html", "r", encoding="utf-8") as f:
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

    @property
    def cached_data_path(self) -> str:
        """Get path to cached data.

        Returns:
            str: Path to cached data.
        """
        return self._cached_data_path

    @cached_data_path.setter
    def cached_data_path(self, cached_data_path: str) -> None:
        """Set path to cached data.

        Args:
            cached_data_path (str): Path to cached data.
        """
        self._cached_data_path = cached_data_path

    @property
    def pretrained_weights_path(self) -> str:
        """Get path to directory containing pretrained weights.

        Returns:
            str: Path to directory containing pretrained weights.
        """
        return self._pretrained_weights_path

    @pretrained_weights_path.setter
    def pretrained_weights_path(self, pretrained_weights_path: str) -> None:
        """Set path to pretrained weights.

        Args:
            pretrained_weights_path (str): Path to pretrained weights.
        """
        self._pretrained_weights_path = pretrained_weights_path

    @property
    def output_path(self) -> str:
        """Get path to output.

        Returns:
            str: Path to output.
        """
        return self._output_path

    @output_path.setter
    def output_path(self, output_path: str) -> None:
        """Set path to output.

        Args:
            output_path (str): Path to output.
        """
        self._output_path = output_path


def _download_git_lfs_file(url) -> bytes:
    """Download a file from git-lfs.

    Args:
        url (str): URL to git-lfs file.

    Returns:
        bytes: File contents.
    """
    ptr_file = requests.get(url, timeout=10)
    ptr = ptr_file.json()
    ptr_sha = ptr["sha"]

    blob_file = requests.get(
        f"https://api.github.com/repos/prob-ml/bliss/git/blobs/{ptr_sha}", timeout=10
    )
    blob = blob_file.json()
    blob_content = blob["content"]
    assert blob["encoding"] == "base64"

    blob_decoded = base64.b64decode(blob_content).decode("utf-8").split("\n")
    sha = blob_decoded[1].split(" ")[1].split(":")[1]
    size = int(blob_decoded[2].split(" ")[1])

    lfs_ptr_file = requests.post(
        "https://github.com/prob-ml/bliss.git/info/lfs/objects/batch",
        headers={
            "Accept": "application/vnd.git-lfs+json",
            # Already added when you pass json=
            # 'Content-type': 'application/json',
        },
        json={
            "operation": "download",
            "transfer": ["basic"],
            "objects": [
                {
                    "oid": sha,
                    "size": size,
                }
            ],
        },
        timeout=10,
    )
    lfs_ptr = lfs_ptr_file.json()
    lfs_ptr_download_url = lfs_ptr["objects"][0]["actions"]["download"]["href"]  # noqa: WPS219

    # Get and write weights to pretrained weights path
    file = requests.get(lfs_ptr_download_url, timeout=10)
    return file.content


def fullcat_to_astropy_table(est_cat: FullCatalog):
    assert list(est_cat.keys()) == [
        "star_log_fluxes",
        "star_fluxes",
        "galaxy_bools",
        "star_bools",
        "galaxy_params",
    ]

    # Convert dictionary of tensors to list of dictionaries
    n = [torch.squeeze(v).shape[0] for v in est_cat.values()][0]  # number of rows
    est_cat_list = [{k: torch.squeeze(v)[i].cpu() for k, v in est_cat.items()} for i in range(n)]

    # Convert to astropy table
    est_cat_table = Table(est_cat_list)
    # Convert all _fluxes columns to u.Quantity
    log_nmgy = u.LogUnit(u.nmgy)
    est_cat_table["star_log_fluxes"] = est_cat_table["star_log_fluxes"] * log_nmgy
    est_cat_table["star_fluxes"] = est_cat_table["star_fluxes"] * u.nmgy

    # Create inner table for galaxy_params
    # Convert list of tensors to list of dictionaries
    galaxy_params_names = [
        "galaxy_flux",
        "galaxy_disk_frac",
        "galaxy_beta_radians",
        "galaxy_disk_q",
        "galaxy_a_d",
        "galaxy_bulge_q",
        "galaxy_a_b",
    ]
    galaxy_params_list = []
    for galaxy_params in est_cat_table["galaxy_params"]:
        galaxy_params_dic = {}
        for i, name in enumerate(galaxy_params_names):
            galaxy_params_dic[name] = galaxy_params[i]
        galaxy_params_dic["galaxy_flux"] = galaxy_params_dic["galaxy_flux"] * u.nmgy
        galaxy_params_dic["galaxy_beta_radians"] = (
            galaxy_params_dic["galaxy_beta_radians"] * u.radian
        )
        galaxy_params_dic["galaxy_a_d"] = galaxy_params_dic["galaxy_a_d"] * u.arcsec
        galaxy_params_dic["galaxy_a_b"] = galaxy_params_dic["galaxy_a_b"] * u.arcsec
        galaxy_params_list.append(galaxy_params_dic)
    galaxy_params_table = Table(galaxy_params_list)
    est_cat_table["galaxy_params"] = galaxy_params_table

    return est_cat_table, galaxy_params_table


def pred_to_astropy_table(pred: Dict[str, Tensor]) -> Table:
    pred.pop("loc")

    # extract parameters from distributions
    dist_params = {}
    for pred_key, pred_dist in pred.items():
        if isinstance(pred_dist, torch.distributions.Categorical):
            probs = rearrange(pred_dist.probs, "b nth ntw ... -> b (nth ntw) ...")
            dist_params[f"{pred_key}_false"] = probs[..., 0]
            dist_params[f"{pred_key}_true"] = probs[..., 1]
        elif (  # noqa: WPS337
            isinstance(
                pred_dist,
                (torch.distributions.Independent, torch.distributions.TransformedDistribution),
            )
        ) and isinstance(pred_dist.base_dist, torch.distributions.Normal):
            dist_params[f"{pred_key}_mean"] = rearrange(
                pred_dist.base_dist.mean, "b nth ntw ... -> b (nth ntw) ..."
            )
            dist_params[f"{pred_key}_std"] = rearrange(
                pred_dist.base_dist.stddev, "b nth ntw ... -> b (nth ntw) ..."
            )
        elif isinstance(pred_dist, (torch.distributions.Normal, torch.distributions.LogNormal)):
            dist_params[f"{pred_key}_mean"] = rearrange(
                pred_dist.mean, "b nth ntw ... -> b (nth ntw) ..."
            )
            dist_params[f"{pred_key}_std"] = rearrange(
                pred_dist.stddev, "b nth ntw ... -> b (nth ntw) ..."
            )
        else:
            raise NotImplementedError(f"Unknown distribution {pred_dist}")

    # convert dictionary of tensors to list of dictionaries
    n = [torch.squeeze(v).shape[0] for v in dist_params.values()][0]  # number of rows
    dist_params_list = [
        {k: torch.squeeze(v)[i].cpu() for k, v in dist_params.items()} for i in range(n)
    ]

    pred_table = Table(dist_params_list)
    # convert values to astropy units
    log_nmgy = u.LogUnit(u.nmgy)
    pred_table["star_log_flux_mean"] = pred_table["star_log_flux_mean"] * log_nmgy
    pred_table["star_log_flux_std"] = pred_table["star_log_flux_std"] * log_nmgy
    pred_table["galsim_flux_mean"] = pred_table["galsim_flux_mean"] * u.nmgy
    pred_table["galsim_flux_std"] = pred_table["galsim_flux_std"] * u.nmgy
    pred_table["galsim_beta_radians_mean"] = pred_table["galsim_beta_radians_mean"] * u.radian
    pred_table["galsim_beta_radians_std"] = pred_table["galsim_beta_radians_std"] * u.radian
    pred_table["galsim_a_d_mean"] = pred_table["galsim_a_d_mean"] * u.arcsec
    pred_table["galsim_a_d_std"] = pred_table["galsim_a_d_std"] * u.arcsec
    pred_table["galsim_a_b_mean"] = pred_table["galsim_a_b_mean"] * u.arcsec
    pred_table["galsim_a_b_std"] = pred_table["galsim_a_b_std"] * u.arcsec

    return pred_table
