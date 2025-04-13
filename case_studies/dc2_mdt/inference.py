import torch
import tqdm
import os
import click

from einops import rearrange
from pathlib import Path

from hydra import initialize, compose
from hydra.utils import instantiate
import pytorch_lightning
from pytorch_lightning.utilities import move_data_to_device

from bliss.surveys.dc2 import DC2DataModule
from bliss.catalog import TileCatalog
from case_studies.dc2_mdt.utils.encoder import DiffusionEncoder


@click.command()
@click.option("--model-tag-name", type=str, required=True, help="the task name (e.g., mdt, simple_net)")
@click.option("--exp-name", type=str, required=True)
@click.option("--exp-check-point-name", type=str, required=True)
@click.option("--cfg-name", type=str, required=True)
@click.option("--cached-data-path", required=True, help="the path to output folder")
@click.option("--cuda-idx", type=int, required=True, help="the index of cuda device")
@click.option("--infer-batch-size", type=int, required=True)
@click.option("--infer-total-iters", type=int, required=True)
def main(model_tag_name, 
         exp_name, 
         exp_check_point_name, 
         cfg_name,
         cached_data_path, 
         cuda_idx, 
         infer_batch_size, 
         infer_total_iters):
    assert model_tag_name in ["mdt", "simple_net", "simple_net_speed", "simple_ar_net", "simple_cond_true_net"]
    model_path = f"../../../bliss_output/DC2_mdt_exp/{exp_name}/checkpoints/{exp_check_point_name}"
    cached_data_path = Path(cached_data_path)
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
    with initialize(config_path="./mdt_config", version_base=None):
        cfg = compose(cfg_name)

    # set the seed
    seed = cfg.train.seed
    print(f"using seed {seed}")
    pytorch_lightning.seed_everything(seed=seed)

    # instantiate some modules
    dc2: DC2DataModule = instantiate(cfg.surveys.dc2)
    dc2.batch_size = infer_batch_size
    dc2.setup(stage="validate")
    dc2_val_dataloader = dc2.val_dataloader()

    bliss_encoder: DiffusionEncoder = instantiate(cfg.encoder).to(device=device)
    pretrained_weights = torch.load(model_path, map_location=device)["state_dict"]
    bliss_encoder.load_state_dict(pretrained_weights)
    bliss_encoder.eval();

    # set inference mode
    if hasattr(bliss_encoder.my_net, "fast_inference_mode"):
        bliss_encoder.my_net.fast_inference_mode = True
        print("enable fast inference mode")
    else:
        print("no fast inference mode")

    # set sampling config
    bliss_encoder.reconfig_sampling(new_sampling_time_steps=500, new_ddim_eta=1.0)

    # grab one batch
    one_batch = next(iter(dc2_val_dataloader))
    one_batch = move_data_to_device(one_batch, device=device)

    target_tile_cat = TileCatalog(one_batch["tile_catalog"])
    target_images = one_batch["images"]
    target_n_sources = target_tile_cat["n_sources"]
    target_tile_cat = target_tile_cat.get_brightest_sources_per_tile(band=2)
    target_locs = target_tile_cat["locs"]
    max_fluxes = bliss_encoder.max_fluxes
    print(f"max_fluxes: {max_fluxes}")
    target_fluxes = target_tile_cat["fluxes"].clamp(max=max_fluxes)
    target_ellipticity = target_tile_cat["ellipticity"]

    # run the inference
    diffusion_cached_file_name = f"{model_tag_name}_posterior_{exp_name}_{exp_check_point_name}_b_{infer_batch_size}_iter_{infer_total_iters}_seed_{seed}.pt"
    if not os.path.isfile(cached_data_path / diffusion_cached_file_name):
        print(f"can't find cached file [{diffusion_cached_file_name}]; rerun the inference")
        init_n_sources = None
        n_sources_list = []
        locs_list = []
        fluxes_list = []
        for i in tqdm.tqdm(range(infer_total_iters)):
            with torch.inference_mode():
                sample_tile_cat = bliss_encoder.sample(one_batch)
            if init_n_sources is None:
                if "n_sources_multi" in sample_tile_cat:
                    init_n_sources = rearrange(sample_tile_cat["n_sources_multi"], "b h w 1 1 -> b h w")
                else:
                    init_n_sources = sample_tile_cat["n_sources"]
            if "n_sources_multi" in sample_tile_cat:
                cur_n_sources = rearrange(sample_tile_cat["n_sources_multi"], "b h w 1 1 -> b h w")
            else:
                cur_n_sources = sample_tile_cat["n_sources"]
            n_sources_list.append(cur_n_sources.cpu())
            locs = sample_tile_cat["locs"][..., 0:1, :]  # (b, h, w, 1, 2)
            locs_list.append(locs.cpu())
            fluxes = sample_tile_cat["fluxes"][..., 0:1, :]  # (b, h, w, 1, 6)
            fluxes_list.append(fluxes.cpu())

        diffusion_result_dict = {
            "init_n_sources": init_n_sources.cpu(),
            "n_sources_list": n_sources_list,
            "locs_list": locs_list,
            "fluxes_list": fluxes_list,
            "target_images": target_images.cpu(),
            "target_n_sources": target_n_sources.cpu(),
            "target_locs": target_locs.cpu(),
            "target_fluxes": target_fluxes.cpu(),
            "target_ellipticity": target_ellipticity.cpu(),
        }
        torch.save(diffusion_result_dict, cached_data_path / diffusion_cached_file_name)
    else:
        print(f"find the cached file [{diffusion_cached_file_name}]; directly use it")
        with open(cached_data_path / diffusion_cached_file_name, "rb") as f:
            diffusion_result_dict = torch.load(f, map_location="cpu")


if __name__ == "__main__":
    main()
