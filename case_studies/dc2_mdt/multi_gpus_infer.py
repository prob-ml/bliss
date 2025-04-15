import torch
import tqdm
import os
import click
import getpass

from einops import rearrange
from pathlib import Path
from termcolor import colored

from hydra import initialize, compose
from hydra.utils import instantiate
import pytorch_lightning
from pytorch_lightning.utilities import move_data_to_device

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from bliss.surveys.dc2 import DC2DataModule
from bliss.catalog import TileCatalog
from case_studies.dc2_mdt.utils.encoder import DiffusionEncoder
from bliss.cached_dataset import DistributedChunkingSampler

def gather_tensor(t: torch.Tensor, put_on_cpu=True):
    world_size = dist.get_world_size()
    t = t.contiguous()
    gathered_tensors = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, t)
    if put_on_cpu:
        return torch.cat(gathered_tensors, dim=0).cpu()
    else:
        return torch.cat(gathered_tensors, dim=0)

@click.command()
@click.option("--model-tag-name", type=str, required=True, help="the task name (e.g., mdt, simple_net)")
@click.option("--exp-name", type=str, required=True)
@click.option("--exp-check-point-name", type=str, required=True)
@click.option("--cfg-name", type=str, required=True)
@click.option("--cached-data-path", required=True, help="the path to output folder")
@click.option("--infer-batch-size", type=int, required=True, help="it's the batch size for each gpu")
@click.option("--infer-total-iters", type=int, required=True)
def main(model_tag_name, 
         exp_name, 
         exp_check_point_name, 
         cfg_name,
         cached_data_path, 
         infer_batch_size, 
         infer_total_iters):
    assert model_tag_name in ["mdt", "simple_net", "simple_net_speed", "simple_ar_net", "simple_cond_true_net"]
    # the relative path fails in a strange way; use absolute path instead
    model_path = f"/home/{getpass.getuser()}/bliss_output/DC2_mdt_exp/{exp_name}/checkpoints/{exp_check_point_name}"
    cached_data_path = Path(cached_data_path)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    def color_print(x, color="green"):
        print(colored(f"[Rank {rank}] " + x, color))
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    color_print(f"using device {torch.cuda.current_device()}; local rank: {local_rank}; world size: {world_size}")

    device = torch.device(torch.cuda.current_device())
    with initialize(config_path="./mdt_config", version_base=None):
        cfg = compose(cfg_name)

    # set the seed
    seed = cfg.train.seed
    pytorch_lightning.seed_everything(seed=seed)

    # instantiate some modules
    dc2: DC2DataModule = instantiate(cfg.surveys.dc2)
    dc2.batch_size = infer_batch_size
    dc2.setup(stage="validate")
    dc2_val_dataloader = dc2.val_dataloader()
    assert isinstance(dc2_val_dataloader.sampler, DistributedChunkingSampler)

    bliss_encoder: DiffusionEncoder = instantiate(cfg.encoder).to(device=device)
    pretrained_weights = torch.load(model_path, map_location=device)["state_dict"]
    bliss_encoder.load_state_dict(pretrained_weights)
    bliss_encoder.eval();
    # set inference mode
    if hasattr(bliss_encoder.my_net, "fast_inference_mode"):
        bliss_encoder.my_net.fast_inference_mode = True
        color_print("enter fast inference mode")
    else:
        color_print("no fast inference mode")
    # set sampling config
    bliss_encoder.reconfig_sampling(new_sampling_time_steps=500, new_ddim_eta=1.0)
    bliss_encoder.my_net = DDP(bliss_encoder.my_net, 
                               device_ids=[torch.cuda.current_device()])

    # grab one batch
    one_batch = next(iter(dc2_val_dataloader))
    one_batch = move_data_to_device(one_batch, device=device)

    target_tile_cat = TileCatalog(one_batch["tile_catalog"])
    target_images = one_batch["images"]
    target_batch_size = target_images.shape[0]
    assert target_batch_size == infer_batch_size
    color_print(f"batch size: {target_batch_size}")
    target_n_sources = target_tile_cat["n_sources"]
    target_tile_cat = target_tile_cat.get_brightest_sources_per_tile(band=2)
    target_locs = target_tile_cat["locs"]
    max_fluxes = bliss_encoder.max_fluxes
    color_print(f"max_fluxes: {max_fluxes}")
    target_fluxes = target_tile_cat["fluxes"].clamp(max=max_fluxes)
    target_ellipticity = target_tile_cat["ellipticity"]

    # run the inference
    diffusion_cached_file_name = f"multi_gpus_{model_tag_name}_" \
                                 f"posterior_{exp_name}_{exp_check_point_name}_" \
                                 f"b_{infer_batch_size}_" \
                                 f"iter_{infer_total_iters}_" \
                                 f"seed_{seed}.pt"
    save_path = cached_data_path / diffusion_cached_file_name
    if not os.path.isfile(save_path):
        color_print(f"can't find cached file [{diffusion_cached_file_name}] at [{str(save_path)}]; rerun the inference", 
                    color="yellow")
        init_n_sources = None
        n_sources_list = []
        locs_list = []
        fluxes_list = []
        if rank == 0:
            iters = tqdm.tqdm(range(infer_total_iters))
        else:
            iters = range(infer_total_iters)
        for i in iters:
            with torch.inference_mode():
                sample_tile_cat = bliss_encoder.sample(one_batch)
            
            if init_n_sources is None:
                if "n_sources_multi" in sample_tile_cat:
                    init_n_sources = rearrange(sample_tile_cat["n_sources_multi"], 
                                               "b h w 1 1 -> b h w")
                else:
                    init_n_sources = sample_tile_cat["n_sources"]
            if "n_sources_multi" in sample_tile_cat:
                cur_n_sources = rearrange(sample_tile_cat["n_sources_multi"], 
                                          "b h w 1 1 -> b h w")
            else:
                cur_n_sources = sample_tile_cat["n_sources"]
            
            n_sources_list.append(gather_tensor(cur_n_sources))
            locs = sample_tile_cat["locs"][..., 0:1, :]  # (b, h, w, 1, 2)
            locs_list.append(gather_tensor(locs))
            fluxes = sample_tile_cat["fluxes"][..., 0:1, :]  # (b, h, w, 1, 6)
            fluxes_list.append(gather_tensor(fluxes))

            diffusion_result_dict = {
                "init_n_sources": gather_tensor(init_n_sources),
                "n_sources_list": n_sources_list,
                "locs_list": locs_list,
                "fluxes_list": fluxes_list,
                "target_images": gather_tensor(target_images),
                "target_n_sources": gather_tensor(target_n_sources),
                "target_locs": gather_tensor(target_locs),
                "target_fluxes": gather_tensor(target_fluxes),
                "target_ellipticity": gather_tensor(target_ellipticity),
            }
        if rank == 0:
            color_print(f"save output to [{str(save_path)}]")
            torch.save(diffusion_result_dict, save_path)
    else:
        color_print(f"find the cached file [{diffusion_cached_file_name}] at [{str(save_path)}]; run nothing")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
