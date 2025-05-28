# pylint: disable=R0801
import pickle
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm


def get_kth_best_ckpt(ckpt_dir: str, k: int = 0):
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    ckpt_files = [Path(f.stem.split("-")[0] + ".ckpt") for f in ckpt_files]
    sorted_files = sorted(ckpt_files, key=lambda f: float(f.stem.split("_")[1]))
    if sorted_files:
        return ckpt_dir / sorted_files[k]

    raise FileExistsError("No ckpt files found in the directory")


@hydra.main(config_path=".", config_name="discrete_eval")
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

    # load bliss trained model - discrete version
    bliss_encoder = instantiate(cfg.encoder).to(device=device)
    print("ckpt_path is {}".format(ckpt_path))
    pretrained_weights = torch.load(ckpt_path, device)["state_dict"]
    bliss_encoder.load_state_dict(pretrained_weights)
    bliss_encoder.eval()

    bliss_discrete_output_path = output_dir / "discrete_mode_metrics_{}thbest.pkl".format(k)
    bliss_discrete_grid_output_path = output_dir / "discrete_grid_metrics_{}thbest.pkl".format(k)

    # compute metrics -- discrete version
    if not bliss_discrete_output_path.exists() or True:
        test_loader = dataset.test_dataloader()
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch["images"] = batch["images"].to(device)
            batch["tile_catalog"] = {
                key: value.to(device) for key, value in batch["tile_catalog"].items()
            }
            batch["psf_params"] = batch["psf_params"].to(device)
            bliss_encoder.update_metrics(batch, batch_idx)
        bliss_mode_out_dict = bliss_encoder.mode_metrics.compute()
        bliss_discrete_out_dict = bliss_encoder.discrete_metrics.compute()

        with open(bliss_discrete_output_path, "wb") as outp:  # Overwrites any existing file.
            pickle.dump(bliss_mode_out_dict, outp, pickle.HIGHEST_PROTOCOL)
        with open(bliss_discrete_grid_output_path, "wb") as outp:  # Overwrites any existing file.
            pickle.dump(bliss_discrete_out_dict, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
