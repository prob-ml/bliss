from pathlib import Path

import torch
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from hydra.utils import instantiate
from tqdm import tqdm
import pickle
from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
import einops


def get_best_ckpt(ckpt_dir: str):
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    sorted_files = sorted(ckpt_files, key=lambda f: float(f.stem.split("_")[1]))
    if sorted_files:
        return sorted_files[0]
    return None


class RedshiftsEncoder(Encoder):
    def __init__(
        self,
        discrete_metrics,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.discrete_metrics = discrete_metrics

    def save_preds(self, batch, batch_idx, use_mode=True, writer=None):
        if writer is None:
            raise ValueError("Provide a writer to save predictions")
        if not use_mode:
            raise NotImplementedError("Only mode predictions are supported")
        
        # # Create file
        # file_path = Path(save_file)
        # if not file_path.exists():
        #     file_path.parent.mkdir(parents=True, exist_ok=True)
        #     # columns = ['z_true', 'z_pred', 'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']
        #     # dtypes = ['float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64']
        #     # empty_df = pd.DataFrame(columns=columns)
        #     # table = pa.Table.from_pandas(empty_df)
        #     # pq.write_table(table, file_path, compression='gzip')
        
        
        target_cat = TileCatalog(batch["tile_catalog"]).get_brightest_sources_per_tile()
        mode_cat = self.sample(batch, use_mode=True)
        matching = self.matcher.match_catalogs(target_cat, mode_cat)

        # Get NLL
        patterns_to_use = torch.randperm(15)[:4] if self.use_checkerboard else (0,)
        history_mask_patterns = self.mask_patterns[patterns_to_use, ...]

        loss_mask_patterns = 1 - history_mask_patterns

        loss = self.compute_masked_nll(batch, history_mask_patterns, loss_mask_patterns) # b x n_tile x n_tile
        loss = einops.rearrange(loss, "b l w -> b l w 1 1") 

        for i in range(target_cat.batch_size): # for each observation in batch
            tcat_matches, ecat_matches = matching[i]
        
            true_red = target_cat["redshifts"][i][tcat_matches].cpu().detach()
            est_red = mode_cat["redshifts"][i][ecat_matches].cpu().detach()
            mags = target_cat["fluxes"][i].unsqueeze(-2)[tcat_matches].cpu().detach()
            nll_true = loss[i][tcat_matches].cpu().detach()

            if len(true_red) == 0 or len(est_red) == 0:
                continue

            batch_data = {
                'z_true': true_red, 
                'z_pred': est_red,
                'nll_true': nll_true, 
                'u_mag': mags[:, 0],
                'g_mag': mags[:, 1],
                'r_mag': mags[:, 2],
                'i_mag': mags[:, 3],
                'z_mag': mags[:, 4],
                'y_mag': mags[:, 5],
            }
            batch_df = pd.DataFrame(batch_data, dtype='float32')
            batch_df = batch_df.reset_index(drop=True)
            batch_table = pa.Table.from_pandas(batch_df, preserve_index=False)
            writer.write_table(batch_table)

            # with pq.ParquetWriter(file_path, batch_table.schema, use_dictionary=True) as writer:
            #     writer.write_table(batch_table)

            # existing_table = pq.read_table(save_file)
            
            # combined_table = pa.concat_tables([existing_table, batch_table])
            # pq.write_table(combined_table, save_file)


        

    def update_metrics(self, batch, batch_idx):
        target_cat = TileCatalog(batch["tile_catalog"]).get_brightest_sources_per_tile()

        # for risk_type in self.discrete_metrics.keys():
        #     mode_cat = self.discrete_sample(batch, use_mode=True, risk_type=risk_type)
        #     matching = self.matcher.match_catalogs(target_cat, mode_cat)
        #     self.discrete_metrics[risk_type].update(target_cat, mode_cat, matching)

        mode_cat = self.sample(batch, use_mode=True)
        matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.mode_metrics.update(target_cat, mode_cat, matching)

        # sample_cat = self.sample(batch, use_mode=False)
        # matching = self.matcher.match_catalogs(target_cat, sample_cat)
        # self.sample_metrics.update(target_cat, sample_cat, matching)

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
        if self.discrete_metrics:  # TODO: should be same as above, not empty dict
            self.discrete_metrics.reset()

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
