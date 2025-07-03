import torch
import tqdm
import os
import click

from pathlib import Path

from hydra import initialize, compose
from hydra.utils import instantiate
import pytorch_lightning
from pytorch_lightning.utilities import move_data_to_device

from bliss.cached_dataset import CachedSimulatedDataModule
from bliss.catalog import TileCatalog
from case_studies.dc2_mdt.utils.rml_encoder import M2MDTRMLEncoder

@click.command()
@click.option("--model-name", type=str, required=True)
@click.option("--model-check-point-name", type=str, required=True)
@click.option("--cached-data-path", required=True, help="the path to output folder")
@click.option("--ddim-sampling-steps", type=int, required=True)
@click.option("--ddim-eta", type=float, required=True)
@click.option("--infer-batch-size", type=int, required=True, help="it's the batch size for each gpu")
@click.option("--infer-total-iters", type=int, required=True)
def main(model_name, model_check_point_name, cached_data_path, ddim_sampling_steps, ddim_eta, infer_batch_size, infer_total_iters):
    model_path = f"../../../bliss_output/{model_name}_m2_mdt_rml_{model_check_point_name}"
    cached_data_path = Path(cached_data_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with initialize(config_path="./m2_mdt_config", version_base=None):
        cfg = compose("m2_mdt_rml_train_config")

    seed = cfg.train.seed
    pytorch_lightning.seed_everything(seed=seed)

    batch_size = infer_batch_size
    m2: CachedSimulatedDataModule = instantiate(cfg.cached_simulator)
    m2.batch_size = batch_size
    m2.setup(stage="validate")
    m2_val_dataloader = m2.val_dataloader()

    my_encoder: M2MDTRMLEncoder = instantiate(cfg.encoder).to(device=device)
    pretrained_weights = torch.load(model_path, map_location=device)["state_dict"]
    my_encoder.load_state_dict(pretrained_weights)
    my_encoder.eval();

    one_batch = next(iter(m2_val_dataloader))
    one_batch = move_data_to_device(one_batch, device=device)

    target_tile_cat = TileCatalog(one_batch["tile_catalog"])
    target_images = one_batch["images"]
    target_n_sources = target_tile_cat["n_sources"]
    target_locs = target_tile_cat["locs"]
    target_fluxes = target_tile_cat["fluxes"]
    max_fluxes = my_encoder.max_fluxes
    print(f"max_fluxes: {max_fluxes}")
    target_fluxes = target_tile_cat["fluxes"].clamp(max=max_fluxes)

    my_encoder.reconfig_sampling(new_sampling_time_steps=ddim_sampling_steps, new_ddim_eta=ddim_eta)

    total_iters = infer_total_iters
    cached_file_name = f"m2_mdt_rml_posterior_" \
                            f"{model_name}_{model_check_point_name}_" \
                            f"ddim_steps_{ddim_sampling_steps}_" \
                            f"eta_{ddim_eta:.1f}_" \
                            f"b_{batch_size}_" \
                            f"iter_{total_iters}_" \
                            f"seed_{seed}.pt"
    save_path = cached_data_path / cached_file_name
    if not os.path.isfile(save_path):
        print("can't find cached file; rerun the inference")
        n_sources_list = []
        locs_list = []
        fluxes_list = []
        my_encoder.my_net.enter_fast_inference()
        for i in tqdm.tqdm(range(total_iters)):
            with torch.no_grad():
                sample_tile_cat = my_encoder.sample(one_batch)
            cur_n_sources = sample_tile_cat["n_sources"]
            n_sources_list.append(cur_n_sources.cpu())
            locs = sample_tile_cat["locs"]  # (b, h, w, 2, 2)
            locs_list.append(locs.cpu())
            fluxes = sample_tile_cat["fluxes"]  # (b, h, w, 2, 6)
            fluxes_list.append(fluxes.cpu())
        my_encoder.my_net.exit_fast_inference()

        diffusion_result_dict = {
            "n_sources_list": n_sources_list,
            "locs_list": locs_list,
            "fluxes_list": fluxes_list,
            "target_images": target_images.cpu(),
            "target_n_sources": target_n_sources.cpu(),
            "target_locs": target_locs.cpu(),
            "target_fluxes": target_fluxes.cpu(),
        }
        torch.save(diffusion_result_dict, save_path)
    else:
        print("find the cached file; run nothing")

if __name__ == "__main__":
    main()
