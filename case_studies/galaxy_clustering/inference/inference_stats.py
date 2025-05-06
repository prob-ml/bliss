# flake8: noqa
import os
import pickle

import numpy as np
import pandas as pd
import torch

DES_DIR = (
    "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles"
)
DES_BANDS = ("g", "r", "i", "z")
DES_SUBDIRS = (d for d in os.listdir(DES_DIR) if d.startswith("DES"))
IMAGE_SIZE = 2560
TILE_SIZE = 256
BATCH_SIZE = 1
NUM_GPUS = 2
TILES_PER_IMG = 16
OUTPUT_DIR = "/data/scratch/des/dr2_detection_output/run_1"
DES_SVA_TILES = pd.read_pickle("/data/scratch/des/sva_map.pickle")
GROUNDTRUTH_PATH = "/data/scratch/des/redmapper_groundtruth"


def convert_to_global_idx(tile_idx, gpu_idx):
    dir_idx = int(NUM_GPUS * (tile_idx // (TILES_PER_IMG / BATCH_SIZE)) + gpu_idx)
    subimage_idx = [(BATCH_SIZE * tile_idx + i) % TILES_PER_IMG for i in range(BATCH_SIZE)]
    return dir_idx, subimage_idx


def convert_to_tile_idx(dir_idx):
    gpu_idx = dir_idx % NUM_GPUS
    tile_starting_idx = (TILES_PER_IMG / BATCH_SIZE) * (dir_idx // NUM_GPUS)
    return int(tile_starting_idx), int(gpu_idx)


def unfolded_memberships(dir_idx):
    memberships = torch.empty((0, 10, 10))
    tile_starting_idx, gpu_idx = convert_to_tile_idx(dir_idx)
    for tile in range(tile_starting_idx, tile_starting_idx + int(TILES_PER_IMG / 2)):
        file = torch.load(
            f"{OUTPUT_DIR}/rank_{gpu_idx}_batchIdx_{tile}_dataloaderIdx_0.pt",
            map_location=torch.device("cpu"),
        )
        memberships = torch.cat((memberships, file["mode_cat"]["membership"].squeeze()), dim=0)
    memberships = torch.repeat_interleave(memberships, repeats=TILE_SIZE, dim=1)
    return torch.repeat_interleave(memberships, repeats=TILE_SIZE, dim=2)


def count_num_clusters():
    num_clusters = {}
    output_filename = "/data/scratch/des/num_clusters.pickle"
    for dir_idx, des_dir in enumerate(DES_SUBDIRS):
        if dir_idx > 10167:
            break
        memberships = unfolded_memberships(dir_idx)
        num_clusters[des_dir] = torch.any(memberships.view(memberships.shape[0], -1), dim=1).sum()

    with open(output_filename, "wb") as handle:
        pickle.dump(num_clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_metrics():
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, des_tile in enumerate(DES_SVA_TILES):
        print(f"Processing tile {i} of {len(DES_SVA_TILES)} ...")
        dir_idx = DES_SUBDIRS.index(des_tile)
        pred_memberships = unfolded_memberships(dir_idx).bool()
        gt_filename = f"/{GROUNDTRUTH_PATH}/{des_tile}_redmapper_groundtruth.npy"
        gt_memberships = torch.from_numpy(np.load(gt_filename))
        unfolded_gt = gt_memberships.unfold(dimension=0, size=2560, step=2480).unfold(
            dimension=1, size=2560, step=2480
        )
        unfolded_gt = unfolded_gt.reshape(-1, 2560, 2560)
        tp += (pred_memberships * unfolded_gt).sum()
        tn += (~pred_memberships * ~unfolded_gt).sum()
        fp += (pred_memberships * ~unfolded_gt).sum()
        fn += (~pred_memberships * unfolded_gt).sum()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1: {f1}")
        print("\n")


def main():
    compute_metrics()
    count_num_clusters()


if __name__ == "__main__":
    main()
