import logging
from pathlib import Path

import einops
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bliss.catalog import TileCatalog
from bliss.encoder.encoder import Encoder

logging.basicConfig(level=logging.INFO)


class RedshiftsEncoder(Encoder):
    def __init__(
        self,
        checkpoint_dir: str,
        plot_dir: str,
        eval_from_checkpoint: bool,
        discrete_metrics,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.discrete_metrics = discrete_metrics
        self.run_name = checkpoint_dir.split("/")[-2]
        self.ckpt_dir = checkpoint_dir if eval_from_checkpoint else None
        self.plot_dir = plot_dir
        self.writer = None  # writer to save predictions to parquet

    def save_preds(self, batch, batch_idx, use_mode=True, writer=None):
        """Called within a test loop to save predictions to disk."""
        if writer is None:
            raise ValueError("Provide a writer to save predictions")
        if not use_mode:
            raise NotImplementedError("Only mode predictions are supported")

        target_cat = TileCatalog(batch["tile_catalog"]).get_brightest_sources_per_tile()
        mode_cat = self.get_mode(batch)
        mean_cat = self.get_mean(batch)
        median_cat = self.get_median(batch)
        matching = self.matcher.match_catalogs(target_cat, mode_cat)

        # Get NLL
        patterns_to_use = torch.randperm(15)[:4] if self.use_checkerboard else (0,)
        history_mask_patterns = self.mask_patterns[patterns_to_use, ...]

        loss_mask_patterns = 1 - history_mask_patterns

        loss = self.compute_masked_nll(
            batch, history_mask_patterns, loss_mask_patterns
        )  # b x n_tile x n_tile
        loss = einops.rearrange(loss, "b l w -> b l w 1 1")

        for i in range(target_cat.batch_size):  # for each observation in batch
            tcat_matches, ecat_matches = matching[i]

            true_red = target_cat["redshifts"][i][tcat_matches].cpu().detach()
            est_red_mode = mode_cat["redshifts"][i][ecat_matches].cpu().detach()
            est_red_mean = mean_cat["redshifts"][i][ecat_matches].cpu().detach()
            est_red_median = median_cat["redshifts"][i][ecat_matches].cpu().detach()

            mags = target_cat["fluxes"][i].unsqueeze(-2)[tcat_matches].cpu().detach()
            nll_true = loss[i][tcat_matches].cpu().detach()

            if len(true_red) == 0 or len(est_red_mode) == 0:
                continue

            batch_data = {
                "z_true": true_red,
                "z_pred_mode": est_red_mode,
                "z_pred_mean": est_red_mean,
                "z_pred_median": est_red_median,
                "nll_true": nll_true,
                "u_mag": mags[:, 0],
                "g_mag": mags[:, 1],
                "r_mag": mags[:, 2],
                "i_mag": mags[:, 3],
                "z_mag": mags[:, 4],
                "y_mag": mags[:, 5],
            }
            batch_df = pd.DataFrame(batch_data, dtype="float32")
            batch_df = batch_df.reset_index(drop=True)
            batch_table = pa.Table.from_pandas(batch_df, preserve_index=False)
            writer.write_table(batch_table)

    def update_metrics(self, batch, batch_idx):
        # Checkpointing metrics must be computable with target/mode cat, matching, and loss.

        # Get target and mode catalogs and matching
        target_cat = TileCatalog(batch["tile_catalog"]).get_brightest_sources_per_tile()
        mode_cat = self.sample(batch, use_mode=True)
        # sample_cat = self.sample(batch, use_mode=False) # unused right now
        matching = self.matcher.match_catalogs(target_cat, mode_cat)

        # Get NLL - TODO: deduplicate from Encoder._compute_loss
        patterns_to_use = torch.randperm(15)[:4] if self.use_checkerboard else (0,)
        history_mask_patterns = self.mask_patterns[patterns_to_use, ...]
        loss_mask_patterns = 1 - history_mask_patterns
        loss = self.compute_masked_nll(
            batch, history_mask_patterns, loss_mask_patterns
        )  # b x n_tile x n_tile
        loss = einops.rearrange(loss, "b l w -> b l w 1 1")

        # Update metrics
        self.mode_metrics.update(target_cat, mode_cat, matching, loss)

    def training_step(self, batch, batch_idx):
        return None

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

    def on_test_epoch_start(self):
        # Optionally load the best checkpoint, else use current weights
        # TODO: this is defunct code due to lines in bliss/main.py; one should change
        if self.ckpt_dir is not None:
            ckpt_dir = Path(self.ckpt_dir)
            ckpt_files = list(ckpt_dir.glob("*.ckpt"))
            sorted_files = sorted(ckpt_files, key=lambda f: float(f.stem.split("_")[1]))
            if sorted_files:
                logging.info(f"Loading checkpoint {sorted_files[0]}")
                ckpt_to_load = sorted_files[0]
            else:
                logging.info(f"No checkpoint found in {ckpt_dir}")
                raise FileExistsError("No ckpt files found in the directory")

            if ckpt_to_load.exists():
                checkpoint = torch.load(ckpt_to_load, map_location=self.device)
                self.load_state_dict(checkpoint["state_dict"])
                self.eval()
            else:
                logging.info(f"Checkpoint {ckpt_to_load} does not exist")
                raise FileExistsError("Specified checkpoint does not exist")

        else:
            logging.info("No checkpoint directory specified, using current weights")

        # Create parquet file for saving predictions
        output_dir = Path(self.plot_dir) / self.run_name
        bliss_output_path = output_dir / "predictions.parquet"
        file_path = Path(bliss_output_path)
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "z_true",
            "z_pred_mode",
            "z_pred_mean",
            "z_pred_median",
            "nll_true",
            "u_mag",
            "g_mag",
            "r_mag",
            "i_mag",
            "z_mag",
            "y_mag",
        ]
        empty_df = pd.DataFrame(columns=columns, dtype="float32")
        dummy_table = pa.Table.from_pandas(empty_df, preserve_index=False)
        self.writer = pq.ParquetWriter(bliss_output_path, dummy_table.schema, use_dictionary=True)

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "test")
        self.update_metrics(batch, batch_idx)
        if self.writer is None:
            logging.warning("Writer not initialized, cannot save predictions")
            return
        else:
            self.save_preds(batch, batch_idx, use_mode=True, writer=self.writer)

    def on_test_epoch_end(self):
        # note: metrics are not reset here, to give notebooks access to them
        self.report_metrics(self.mode_metrics, "test/mode", show_epoch=False)
        if self.sample_metrics is not None:
            self.report_metrics(self.sample_metrics, "test/sample", show_epoch=False)
        if self.writer is not None:
            self.writer.close()

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

    def get_mode(self, batch):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.get_mode(x_cat_marginal)

    def get_mean(self, batch):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.get_mean(x_cat_marginal)

    def get_median(self, batch):
        _, x_cat_marginal = self.get_features_and_parameters(batch)
        return self.var_dist.get_median(x_cat_marginal)
