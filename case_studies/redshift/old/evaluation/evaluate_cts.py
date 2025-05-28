# pylint: disable=R0801
import pickle
from pathlib import Path

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm


def get_kth_best_ckpt(ckpt_dir: str, k: int = 0):
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    sorted_files = sorted(ckpt_files, key=lambda f: float(f.stem.split("_")[1]))
    if sorted_files:
        return sorted_files[k]

    raise FileExistsError("No ckpt files found in the directory")


@hydra.main(config_path=".", config_name="continuous_eval")
def main(cfg: DictConfig):
    k = 0
    output_dir = cfg.paths.plot_dir
    ckpt_dir = cfg.paths.ckpt_dir
    run_name = ckpt_dir.split("/")[-2]

    output_dir = Path(output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = get_kth_best_ckpt(ckpt_dir, k)
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # set up testing dataset
    dataset = instantiate(cfg.train.data_source)
    dataset.setup("test")

    # load bliss trained model - continuous version
    bliss_encoder = instantiate(cfg.encoder).to(device=device)
    pretrained_weights = torch.load(ckpt_path, device)["state_dict"]
    bliss_encoder.load_state_dict(pretrained_weights)
    bliss_encoder.eval()

    # load bliss trained model - continuous version
    bliss_output_path = output_dir / "cts_mode_predictions.parquet"
    file_path = Path(bliss_output_path)
    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # create empty parquet file
    columns = [
        "z_true",
        "z_pred",
        "nll_true",
        "u_mag",
        "g_mag",
        "r_mag",
        "i_mag",
        "z_mag",
        "y_mag",
    ]
    dtypes = [
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
    ]
    empty_df = pd.DataFrame(columns=columns, dtype="float32")
    # empty_df = empty_df.astype({col: dtype for col, dtype in zip(columns, dtypes)})
    dummy_table = pa.Table.from_pandas(empty_df, preserve_index=False)

    # compute metrics -- continuous version
    if not bliss_output_path.exists() or True:
        test_loader = dataset.test_dataloader()
        with pq.ParquetWriter(bliss_output_path, dummy_table.schema, use_dictionary=True) as writer:
            # writer.write_table(batch_table)
            for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                batch["images"] = batch["images"].to(device)
                batch["tile_catalog"] = {
                    key: value.to(device) for key, value in batch["tile_catalog"].items()
                }
                batch["psf_params"] = batch["psf_params"].to(device)
                bliss_encoder.save_preds(batch, batch_idx, use_mode=True, writer=writer)
        # bliss_out_dict = bliss_encoder.mode_metrics.compute()

        # with open(bliss_output_path, "wb") as outp:  # Overwrites any existing file.
        #     pickle.dump(bliss_out_dict, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
