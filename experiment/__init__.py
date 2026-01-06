"""Replace the following paths with where you want to save figures, datasets, etc."""

from pathlib import Path

SEED = 52

DATASETS_DIR = Path("/nfs/turbo/lsa-regier/scratch/ismael/datasets/")
FIGURE_DIR = Path("/home/imendoza/bliss/experiment/figures/")
MODELS_DIR = Path("/home/imendoza/bliss/experiment/models/")

# figure creation might store intermediate files here
CACHE_DIR = Path("/nfs/turbo/lsa-regier/scratch/ismael/cache/")

# metadata for models saved by pytorch lightning
TORCH_DIR = Path("/nfs/turbo/lsa-regier/scratch/ismael/out/")
