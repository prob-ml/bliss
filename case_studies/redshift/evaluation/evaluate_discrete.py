# pylint: disable=R0801
import pickle
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm


def get_best_ckpt(ckpt_dir: str):
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    sorted_files = sorted(ckpt_files, key=lambda f: float(f.stem.split("_")[1]))
    if sorted_files:
        return sorted_files[0]
    raise FileExistsError("No ckpt files found in the directory")


@hydra.main(config_path=".", config_name="discrete_eval")
def main(cfg: DictConfig):
    output_dir = cfg.paths.plot_dir
    ckpt_dir = cfg.paths.ckpt_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = get_best_ckpt(ckpt_dir)
    device = torch.device("cpu")

    # set up testing dataset
    dataset = instantiate(cfg.train.data_source)
    dataset.setup("test")

    # load bliss trained model - discrete version
    bliss_encoder = instantiate(cfg.encoder).to(device=device)
    pretrained_weights = torch.load(ckpt_path, device)["state_dict"]
    bliss_encoder.load_state_dict(pretrained_weights)
    bliss_encoder.eval()

    bliss_discrete_output_path = output_dir / "discrete_mode_metrics.pkl"
    bliss_discrete_grid_output_path = output_dir / "discrete_grid_metrics.pkl"

    # compute metrics -- discrete version
    if not bliss_discrete_output_path.exists():
        test_loader = dataset.test_dataloader()
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch["images"] = batch["images"].to(device)
            bliss_encoder.update_metrics(batch, batch_idx)
        bliss_mode_out_dict = bliss_encoder.mode_metrics.compute()
        bliss_discrete_out_dict = bliss_encoder.discrete_metrics.compute()

        with open(bliss_discrete_output_path, "wb") as outp:  # Overwrites any existing file.
            pickle.dump(bliss_mode_out_dict, outp, pickle.HIGHEST_PROTOCOL)
        with open(bliss_discrete_grid_output_path, "wb") as outp:  # Overwrites any existing file.
            pickle.dump(bliss_discrete_out_dict, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
