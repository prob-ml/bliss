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

from bliss.cached_dataset import CachedSimulatedDataModule
from bliss.catalog import TileCatalog
from case_studies.dc2_mdt.utils.encoder import M2DiffusionEncoder
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
@click.option("--model-tag-name", type=str, required=True)
@click.option("--exp-check-point-path", type=str, required=True)
@click.option("--cfg-path", type=str, required=True)
@click.option("--cached-data-path", required=True, help="the path to output folder")
@click.option("--infer-batch-size", type=int, required=True, help="it's the batch size for each gpu")
@click.option("--infer-total-iters", type=int, required=True)
def main(model_tag_name,
         exp_check_point_path, 
         cfg_path,
         cached_data_path, 
         infer_batch_size, 
         infer_total_iters):
    # the relative path fails in a strange way; use absolute path instead
    exp_check_point_path = Path(exp_check_point_path)
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
    cfg_path = Path(cfg_path)
    with initialize(config_path=str(cfg_path.parent), version_base=None):
        cfg = compose(cfg_path.name)

    # set the seed
    seed = cfg.train.seed
    pytorch_lightning.seed_everything(seed=seed)

    # instantiate some modules
    m2: CachedSimulatedDataModule = instantiate(cfg.cached_simulator)
    m2.batch_size = infer_batch_size
    m2.setup(stage="validate")
    m2_val_dataloader = m2.val_dataloader()
    assert isinstance(m2_val_dataloader.sampler, DistributedChunkingSampler)

    my_encoder: M2DiffusionEncoder = instantiate(cfg.encoder).to(device=device)
    pretrained_weights = torch.load(exp_check_point_path, map_location=device)["state_dict"]
    my_encoder.load_state_dict(pretrained_weights)
    my_encoder.eval();
    # set inference mode
    if hasattr(my_encoder.my_net, "fast_inference_mode"):
        my_encoder.my_net.fast_inference_mode = True
        color_print("enter fast inference mode")
    else:
        color_print("no fast inference mode")
    # set sampling config
    my_encoder.reconfig_sampling(new_sampling_time_steps=500, new_ddim_eta=1.0)
    my_encoder.my_net = DDP(my_encoder.my_net, 
                            device_ids=[torch.cuda.current_device()])

    # grab one batch
    one_batch = next(iter(m2_val_dataloader))
    one_batch = move_data_to_device(one_batch, device=device)

    target_tile_cat = TileCatalog(one_batch["tile_catalog"])
    target_images = one_batch["images"]
    target_batch_size = target_images.shape[0]
    assert target_batch_size == infer_batch_size
    color_print(f"batch size: {target_batch_size}")
    target_n_sources = target_tile_cat["n_sources"]
    target_locs = target_tile_cat["locs"]
    max_fluxes = my_encoder.max_fluxes
    color_print(f"max_fluxes: {max_fluxes}")
    target_fluxes = target_tile_cat["fluxes"].clamp(max=max_fluxes)

    # run the inference
    exp_name, exp_check_point_name = exp_check_point_path.parts[-3], exp_check_point_path.parts[-1]
    diffusion_cached_file_name = f"{model_tag_name}_" \
                                 f"posterior_{exp_name}_{exp_check_point_name}_" \
                                 f"b_{infer_batch_size}_" \
                                 f"iter_{infer_total_iters}_" \
                                 f"seed_{seed}.pt"
    save_path = cached_data_path / diffusion_cached_file_name
    if not os.path.isfile(save_path):
        color_print(f"can't find cached file [{diffusion_cached_file_name}] at [{str(save_path)}]; rerun the inference", 
                    color="yellow")
        n_sources_list = []
        locs_list = []
        fluxes_list = []
        if rank == 0:
            iters = tqdm.tqdm(range(infer_total_iters))
        else:
            iters = range(infer_total_iters)
        for i in iters:
            with torch.inference_mode():
                sample_tile_cat = my_encoder.sample(one_batch)
            cur_n_sources = sample_tile_cat["n_sources"]
            n_sources_list.append(gather_tensor(cur_n_sources))
            locs = sample_tile_cat["locs"]  # (b, h, w, 2, 2)
            locs_list.append(gather_tensor(locs))
            fluxes = sample_tile_cat["fluxes"]  # (b, h, w, 2, 6)
            fluxes_list.append(gather_tensor(fluxes))

            diffusion_result_dict = {
                "n_sources_list": n_sources_list,
                "locs_list": locs_list,
                "fluxes_list": fluxes_list,
                "target_images": gather_tensor(target_images),
                "target_n_sources": gather_tensor(target_n_sources),
                "target_locs": gather_tensor(target_locs),
                "target_fluxes": gather_tensor(target_fluxes),
            }
        if rank == 0:
            color_print(f"save output to [{str(save_path)}]")
            torch.save(diffusion_result_dict, save_path)
    else:
        color_print(f"find the cached file [{diffusion_cached_file_name}] at [{str(save_path)}]; run nothing")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
