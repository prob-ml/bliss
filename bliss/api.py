from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, TypeAlias

import torch
from astropy import units as u  # noqa: WPS347
from astropy.table import Table
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor

from bliss.catalog import FullCatalog
from bliss.conf.igs import base_config
from bliss.generate import generate as _generate
from bliss.predict import predict_sdss as _predict_sdss
from bliss.surveys.sdss import SDSSDownloader
from bliss.train import train as _train
from bliss.utils.download_utils import download_git_lfs_file

SurveyType: TypeAlias = Literal["decals", "hst", "lsst", "sdss"]


class BlissClient:
    """Client for interacting with the BLISS API."""

    def __init__(self, cwd: str):
        self._cwd = cwd
        self.base_cfg = base_config()
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

        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        _generate(cfg)

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
        cfg.training.weight_save_path = cfg.paths.output + f"/{weight_save_path}"
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
            **kwargs: Keyword arguments to override default configuration values.
        """
        cfg = OmegaConf.create(self.base_cfg)
        cfg.training.use_cached_simulator = True
        # apply overrides
        cfg.training.weight_save_path = cfg.paths.output + f"/{weight_save_path}"
        cfg.cached_simulator.train_n_batches = train_n_batches
        cfg.cached_simulator.batch_size = batch_size
        cfg.cached_simulator.val_split_file_idxs = val_split_file_idxs
        if pretrained_weights_filename is not None:
            cfg.training.pretrained_weights = (
                cfg.paths.pretrained_models + f"/{pretrained_weights_filename}"
            )
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        _train(cfg)

    def load_survey(self, survey: SurveyType, run, camcol, field, download_dir: str):
        SDSSDownloader(run, camcol, field, self.cwd + f"/{download_dir}").download_all()
        # assert files downloaded at download_dir

    def predict_sdss(
        self,
        weight_save_path: str,
        **kwargs,
    ) -> Tuple[FullCatalog, Table, Table, Table]:
        """Predict on SDSS images.

        Args:
            weight_save_path (str): Path to directory after cwd where trained model
                weights are stored.
            **kwargs: Keyword arguments to override default configuration values.

        Returns:
            Tuple[FullCatalog, Table, Table]: Tuple of predicted catalog, astropy.table
            of predicted catalog, and astropy.table of predicted galaxy_params.
        """
        cfg = OmegaConf.create(self.base_cfg)
        # apply overrides
        cfg.predict.weight_save_path = cfg.paths.output + f"/{weight_save_path}"
        for k, v in kwargs.items():
            OmegaConf.update(cfg, k, v)

        est_cat, _, _, _, pred = _predict_sdss(cfg)
        est_cat_table, galaxy_params_table = fullcat_to_astropy_table(est_cat)
        pred_table = pred_to_astropy_table(pred)
        return est_cat, est_cat_table, galaxy_params_table, pred_table

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


def fullcat_to_astropy_table(est_cat: FullCatalog):
    assert list(est_cat.keys()) == [
        "star_log_fluxes",
        "star_fluxes",
        "source_type",
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
