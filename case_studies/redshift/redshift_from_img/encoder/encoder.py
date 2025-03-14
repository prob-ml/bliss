import torch

from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder
from hydra import compose, initialize
import os
import pickle
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def get_best_ckpt(ckpt_dir: str):
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    sorted_files = sorted(ckpt_files, key=lambda f: float(f.stem.split("_")[1]))
    if sorted_files:
        return sorted_files[0]
    else:
        return None
    raise FileExistsError("No ckpt files found in the directory")

class RedshiftsEncoder(Encoder):
    def __init__(
        self,
        discrete_metrics,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.discrete_metrics = discrete_metrics

    def update_metrics(self, batch, batch_idx):
        target_cat = TileCatalog(batch["tile_catalog"]).get_brightest_sources_per_tile()

        for risk_type in self.discrete_metrics.keys():
            mode_cat = self.discrete_sample(batch, use_mode=True, risk_type=risk_type)
            matching = self.matcher.match_catalogs(target_cat, mode_cat)
            self.discrete_metrics[risk_type].update(target_cat, mode_cat, matching)

        mode_cat = self.sample(batch, use_mode=True)
        matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.mode_metrics.update(target_cat, mode_cat, matching)

        sample_cat = self.sample(batch, use_mode=False)
        matching = self.matcher.match_catalogs(target_cat, sample_cat)
        self.sample_metrics.update(target_cat, sample_cat, matching)

    def on_validation_epoch_end(self):
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/2529
        self.mode_metrics = self.mode_metrics.to(self.device)
        self.sample_metrics = self.sample_metrics.to(self.device)
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.report_metrics(self.sample_metrics, "val/sample", show_epoch=True)

        # Explicitly reset metrics trial
        self.mode_metrics.reset()
        if self.sample_metrics is not None:
            self.sample_metrics.reset()
        if self.discrete_metrics: # should be same as above, not empty dict
            self.discrete_metrics.reset()

        # del self.mode_metrics, self.sample_metrics
        # if self.current_epoch % 5 == 0:
        #     if self.logger.version == 'continuous':
        #         # Compute continuous metrics
        #         cfg = compose("redshift_continuous.yaml")
        #         OmegaConf.resolve(cfg)
        #         cfg.surveys.dc2.batch_size = 4
        #         cfg.train.data_source.batch_size = 4
        #         output_dir = cfg.paths.plot_dir
        #         ckpt_dir = cfg.paths.ckpt_dir

        #         output_dir = Path(output_dir)
        #         output_dir.mkdir(parents=True, exist_ok=True)

        #         ckpt_path = get_best_ckpt(ckpt_dir)
        #         if not ckpt_path:
        #             return
        #         device = self.device

        #         # set up testing dataset
        #         dataset = instantiate(cfg.train.data_source)
        #         dataset.setup("test")

        #         # load bliss trained model - continuous version
        #         bliss_encoder = instantiate(cfg.encoder).to(device=device)
        #         pretrained_weights = torch.load(ckpt_path, device)["state_dict"]
        #         bliss_encoder.load_state_dict(pretrained_weights)
        #         bliss_encoder.eval()

        #         # load bliss trained model - continuous version
        #         bliss_output_path = output_dir / "cts_mode_metrics_{}.pkl".format(self.current_epoch)
        #         test_loader = dataset.test_dataloader()
        #         for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        #             batch["images"] = batch["images"].to(device)
        #             batch["tile_catalog"] = {key: value.to(device) for key, value in batch["tile_catalog"].items()}
        #             batch["psf_params"] = batch["psf_params"].to(device)
        #             bliss_encoder.update_metrics(batch, batch_idx)
        #         bliss_out_dict = bliss_encoder.mode_metrics.compute()
        #         del test_loader
        #         del bliss_encoder
        #         del dataset
        #         del pretrained_weights

        #         with open(bliss_output_path, "wb") as outp:  # Overwrites any existing file.
        #             pickle.dump(bliss_out_dict, outp, pickle.HIGHEST_PROTOCOL)

        #     if self.logger.version == 'discrete':
        #         cfg = compose("redshift_discrete.yaml")
        #         OmegaConf.resolve(cfg)
        #         cfg.surveys.dc2.batch_size = 4
        #         cfg.train.data_source.batch_size = 4
        #         output_dir = cfg.paths.plot_dir
        #         ckpt_dir = cfg.paths.ckpt_dir

        #         output_dir = Path(output_dir)
        #         output_dir.mkdir(parents=True, exist_ok=True)

        #         ckpt_path = get_best_ckpt(ckpt_dir)
        #         if not ckpt_path:
        #             return
        #         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #         # set up testing dataset
        #         dataset = instantiate(cfg.train.data_source)
        #         dataset.setup("test")

        #         # load bliss trained model - discrete version
        #         bliss_encoder = instantiate(cfg.encoder).to(device=device)
        #         pretrained_weights = torch.load(ckpt_path, device)["state_dict"]
        #         bliss_encoder.load_state_dict(pretrained_weights)
        #         bliss_encoder.eval()

        #         bliss_discrete_output_path = output_dir / "discrete_mode_metrics_{}.pkl".format(self.current_epoch)
        #         bliss_discrete_grid_output_path = output_dir / "discrete_grid_metrics_{}.pkl".format(self.current_epoch)

        #         # compute metrics -- discrete version
                
        #         test_loader = dataset.test_dataloader()
        #         for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        #             batch["images"] = batch["images"].to(device)
        #             batch["tile_catalog"] = {key: value.to(device) for key, value in batch["tile_catalog"].items()}
        #             batch["psf_params"] = batch["psf_params"].to(device)
        #             bliss_encoder.update_metrics(batch, batch_idx)
        #         bliss_mode_out_dict = bliss_encoder.mode_metrics.compute()
        #         bliss_discrete_out_dict = bliss_encoder.discrete_metrics.compute()

        #         with open(bliss_discrete_output_path, "wb") as outp:  # Overwrites any existing file.
        #             pickle.dump(bliss_mode_out_dict, outp, pickle.HIGHEST_PROTOCOL)
        #         with open(bliss_discrete_grid_output_path, "wb") as outp:  # Overwrites any existing file.
        #             pickle.dump(bliss_discrete_out_dict, outp, pickle.HIGHEST_PROTOCOL)

            

    def get_features_and_parameters(self, batch):
        batch = (
            batch
            if isinstance(batch, dict)
            else {"images": batch, "background": torch.zeros_like(batch)}
        )
        x_features = self.get_features(batch)
        batch_size, _n_features, ht, wt = x_features.shape[0:4]
        pattern_to_use = (0,)  # no checkerboard
        mask_pattern = self.mask_patterns[pattern_to_use, ...]
        est_cat = None
        history_mask = mask_pattern.repeat([batch_size, ht // 2, wt // 2])
        x_color_context = self.make_color_context(est_cat, history_mask)
        x_features_color = torch.cat((x_features, x_color_context), dim=1)
        x_cat_marginal = self.detect_first(x_features_color)

        return x_features_color, x_cat_marginal

    def sample(self, batch, use_mode=True):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.sample(x_cat_marginal, use_mode=use_mode)

    def discrete_sample(self, batch, use_mode=True, risk_type=None):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.discrete_sample(x_cat_marginal, use_mode=use_mode, risk_type=risk_type)
