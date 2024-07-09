import case_studies.galaxy_clustering.cached_dataset as cachedDataset
import argparse

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as distributed


from hydra import initialize, compose
from hydra.utils import instantiate

DES_DATAPATH = "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles"
DES_DIRECTORIES = f"{DES_DATAPATH}"

import os

def load_model(model_path, device):
    with initialize(config_path=".", version_base=None):
        cfg = compose("config")
    encoder = instantiate(cfg.predict.encoder)
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    enc_state_dict = enc_state_dict["state_dict"]
    encoder.load_state_dict(enc_state_dict).to(device)
    #print(os.path.exists(model_path))
    #model = torch.load(model_path)["state_dict"].to(device)
    #model.eval()
    return encoder

def inference(rank, world_size, cached_data_path, model_path, gpu_ids):
    #distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    des_dataset = cachedDataset.DESDataset(DES_DIRECTORIES, 64)
    sampler = cachedDataset.DistributedDESSampler(
                                                des_dataset,
                                                num_replicas=world_size,
                                                rank=rank,
                                                shuffle=False
                                                )
    dataloader = DataLoader(des_dataset, sampler=sampler, batch_size=32)
    device = torch.device(f'cuda:{gpu_ids[rank]}')
    model = load_model(model_path, device=device)

    rank_indices = []
    for batch_indices in dataloader.batch_sampler:
        rank_indices.extend(batch_indices)

    print(f"Rank {rank}: Indices")
    for batch in dataloader:
        batch.to(device)
        print(f"Rank {rank}: {rank_indices[:100]}")
        print(rank, batch.shape)

        with torch.no_grad():
            output = model(batch)
        break

    distributed.destroy_process_group()


    
def main():
    parser = argparse.ArgumentParser(description='Distributed Inference')
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='List of GPU indices to use')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    args = parser.parse_args()

    gpu_ids = args.gpus
    print(gpu_ids)
    model_path = args.model_path
    data_path = args.data_path
    world_size = len(gpu_ids)

    mp.spawn(inference,
             args=(world_size, data_path, model_path, gpu_ids),
             nprocs=world_size,
             join=True)
    
if __name__ == "__main__":
    main()