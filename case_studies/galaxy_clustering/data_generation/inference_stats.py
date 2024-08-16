import os
import pickle

import torch

DES_DIR = (
    "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles"
)
DES_BANDS = ("g", "r", "i", "z")
DES_SUBDIRS = (d for d in os.listdir(DES_DIR) if d.startswith("DES"))
OUTPUT_DIR = "/data/scratch/des/dr2_detection_output/run_1"


def convert_to_global_idx(tile_idx, gpu_idx):
    num_gpus = 2
    tiles_per_img = 64
    batch_size = 2
    dir_idx = int(num_gpus * (tile_idx // (tiles_per_img / batch_size)) + gpu_idx)
    subimage_idx = [(batch_size * tile_idx + i) % tiles_per_img for i in range(batch_size)]
    return dir_idx, subimage_idx


def convert_to_tile_idx(dir_idx):
    num_gpus = 2
    tiles_per_img = 64
    batch_size = 2
    gpu_idx = dir_idx % num_gpus
    tile_starting_idx = (tiles_per_img / batch_size) * (dir_idx // num_gpus)
    return int(tile_starting_idx), int(gpu_idx)


def count_num_clusters(dir_idx):
    memberships = torch.empty((0, 10, 10))
    tile_starting_idx, gpu_idx = convert_to_tile_idx(dir_idx)
    for tile in range(tile_starting_idx, tile_starting_idx + 32):
        file = torch.load(
            f"{OUTPUT_DIR}/rank_{gpu_idx}_batchIdx_{tile}_dataloaderIdx_0.pt",
            map_location=torch.device("cpu"),
        )
        memberships = torch.cat((memberships, file["mode_cat"]["membership"].squeeze()), dim=0)
    memberships = torch.repeat_interleave(memberships, repeats=128, dim=1)
    memberships = torch.repeat_interleave(memberships, repeats=128, dim=2)
    return torch.any(memberships.view(memberships.shape[0], -1), dim=1).sum()


def main():
    num_clusters = {}
    output_filename = "/data/scratch/des/num_clusters.pickle"
    for dir_idx, des_dir in enumerate(DES_SUBDIRS):
        if dir_idx > 10167:
            break
        num_clusters[des_dir] = count_num_clusters(dir_idx)

    with open(output_filename, "wb") as handle:
        pickle.dump(num_clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
