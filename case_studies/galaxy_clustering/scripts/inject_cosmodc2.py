import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random
import glob
import os

# === Config ===
tiles_csv = "/home/hughwang/bliss/bliss/case_studies/galaxy_clustering/notebooks/all_false_tiles.csv"   # CSV with column 'tile_path'
clusters_dir = "/home/hughwang/bliss/bliss/output_ori" # folder containing cluster sets
output_dir = "/home/hughwang/bliss/bliss/injected_tiles"
os.makedirs(output_dir, exist_ok=True)

# === Functions ===
def load_fits(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float32)

def stack_cluster(g, r, i, z):
    return np.stack([g, r, i, z], axis=0)  # [4, H, W]

def downsample_cluster(cluster_np):
    tensor = torch.from_numpy(cluster_np).unsqueeze(0).float()
    pooled = F.avg_pool2d(tensor, kernel_size=2, stride=2)
    return pooled.squeeze(0)

def inject_random_location(tile_img, cluster_patch, membership, method="add"):
    _, H, W = tile_img.shape
    _, h, w = cluster_patch.shape
    assert h <= H and w <= W, "Cluster larger than tile"

    y0 = random.randint(0, H - h)
    x0 = random.randint(0, W - w)

    if method == "add":
        tile_img[:, y0:y0+h, x0:x0+w] += cluster_patch
    elif method == "overwrite":
        tile_img[:, y0:y0+h, x0:x0+w] = cluster_patch
    else:
        raise ValueError("Method must be 'add' or 'overwrite'")

    # Update membership (19x19 grid)
    patch_size = 256
    y_start = y0 // patch_size
    y_end   = (y0 + h - 1) // patch_size
    x_start = x0 // patch_size
    x_end   = (x0 + w - 1) // patch_size
    for row in range(y_start, y_end + 1):
        for col in range(x_start, x_end + 1):
            if 0 <= row < 19 and 0 <= col < 19:
                membership[0, row, col] = True

    return tile_img, membership, y0, x0, h, w

def choose_random_cluster(clusters_dir):
    """
    Randomly choose one cluster set (g,r,i,z FITS) from the folder.
    Matches files like: cluster_halo_xxx_healpix_xxx_{g,r,i,z}.fits
    """
    files = os.listdir(clusters_dir)

    # find base names (strip the _g.fits / _r.fits / _i.fits / _z.fits suffix)
    bases = set()
    for file in files:
        if file.endswith(("_g.fits", "_r.fits", "_i.fits", "_z.fits")):
            base = file.rsplit("_", 1)[0]   # everything before "_g.fits"
            bases.add(base)

    if not bases:
        raise RuntimeError(f"No cluster sets found in {clusters_dir}. Check filenames/extensions.")

    base = random.choice(list(bases))

    g = load_fits(os.path.join(clusters_dir, f"{base}_g.fits"))
    r = load_fits(os.path.join(clusters_dir, f"{base}_r.fits"))
    i = load_fits(os.path.join(clusters_dir, f"{base}_i.fits"))
    z = load_fits(os.path.join(clusters_dir, f"{base}_z.fits"))

    return stack_cluster(g, r, i, z), base


def visualize_and_save(tile_img, x0, y0, w, h, out_png):
    g = tile_img[0].numpy()
    r = tile_img[1].numpy()
    i = tile_img[2].numpy()
    rgb = make_lupton_rgb(i, r, g)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin="lower")
    rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title("Cluster Injected")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# === Main Loop ===
base_dir = Path("/scratch/regier_root/regier0/hughwang/realCombined_dr2")  # folder with .pt tiles

df = pd.read_csv(tiles_csv)
for idx, row in df.iterrows():
    tile_name = row["tile_name"]  # e.g., "DES0132-5831"

    # build the actual file path
    tile_path = base_dir / f"file_data_{tile_name}_imagesize_9728_tilesize_512.pt"
    out_pt = os.path.join(output_dir, f"{tile_name}_injected.pt")
    out_png = os.path.join(output_dir, f"{tile_name}_injected.png")

    # Choose a random cluster
    cluster_np, cluster_id = choose_random_cluster(clusters_dir)
    cluster_patch = downsample_cluster(cluster_np)

    # Load tile
    tile_data = torch.load(tile_path, map_location="cpu")
    tile_img = tile_data["images"]
    membership = tile_data["tile_catalog"]["membership"]

    # Inject
    tile_img, membership, y0, x0, h, w = inject_random_location(tile_img, cluster_patch, membership)

    # Save updated .pt
    torch.save({"images": tile_img, "tile_catalog": {"membership": membership}}, out_pt)
    print(f"Saved injected tile {tile_name} with cluster {cluster_id} → {out_pt}")

    # Save visualization
    visualize_and_save(tile_img, x0, y0, w, h, out_png)
    print(f"Saved visualization → {out_png}")
