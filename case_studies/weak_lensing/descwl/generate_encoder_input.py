import logging
import os
import time
from datetime import datetime, timedelta

import torch
import yaml

from bliss.catalog import TileCatalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    logger.info("tqdm not installed. Install with: pip install tqdm")

    def tqdm(iterable, desc="", total=None):  # noqa: WPS440
        return iterable


def compress_shear(shear_tensor):
    return shear_tensor.mean(dim=2)


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    hours = seconds / 3600
    return f"{hours:.1f}h"


def convert_save_images_catalogs(
    images,
    catalogs,
    setting="meta",
    shear_feature="varying",
):
    """Convert generated catalogs to TileCatalog object and combine with generated images.

    Args:
        images: torch.Tensor
            Loaded image tensors with shape [batch_size, num_of_bands, 2048, 2048]
        catalogs: dict
            {
                "locs": (N, n_tiles_per_side, n_tiles_per_side, max_num_of_sources, 2),
                "n_sources": (N, n_tiles_per_side, n_tiles_per_side),
                "shear_1": (N, n_tiles_per_side, n_tiles_per_side, max_num_of_sources, 1),
                "shear_2": (N, n_tiles_per_side, n_tiles_per_side, max_num_of_sources, 1),
            }
        setting: str
            Name of the simulation setting
        shear_feature: str
            Whether shear is "constant" or "varying"
    """
    logger.info(f"Starting processing for {setting}_{shear_feature}_shear...")
    logger.info(f"Total images to process: {len(images)}")

    overall_start_time = time.time()

    logger.info("Generating TileCatalog object...")
    tc = TileCatalog(catalogs)
    logger.info(f"TileCatalog created in {format_time(time.time() - overall_start_time)}")

    save_folder = (
        f"/nfs/turbo/lsa-regier/scratch/taodingr/"
        f"weak_lensing_img2048_{setting}_{shear_feature}_shear"
    )
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"Images will be saved to: {save_folder}")

    processing_times = []

    for idx, image in enumerate(tqdm(images, desc="Processing images", unit="img")):
        start = time.time()

        tile_catalog = {
            "shear_1": compress_shear(tc["shear_1"][idx]),
            "shear_2": compress_shear(tc["shear_2"][idx]),
        }

        dict_data = {
            "images": image.clone(),
            "tile_catalog": tile_catalog,
        }
        torch.save([dict_data], f"{save_folder}/dataset_{idx}_size_1.pt")

        processing_times.append(time.time() - start)

        if idx == 0 or (idx + 1) % 10 == 0:
            avg_time = sum(processing_times) / len(processing_times)
            remaining = len(images) - (idx + 1)
            eta = datetime.now() + timedelta(seconds=avg_time * remaining)
            logger.info(
                f"Processed {idx + 1}/{len(images)} | "
                f"Avg time: {format_time(avg_time)} | "
                f"ETA: {eta.strftime('%H:%M:%S')} ({format_time(avg_time * remaining)} remaining)",
            )

    total_time = time.time() - overall_start_time
    logger.info("Processing complete!")
    logger.info(f"Total time: {format_time(total_time)}")
    logger.info(f"Average time per image: {format_time(total_time / len(images))}")
    logger.info(f"Images saved to: {save_folder}")


def main():
    logger.info("=" * 60)
    logger.info(
        f"Starting weak lensing data processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )
    logger.info("=" * 60)

    start = time.time()

    try:
        with open("encoder_input_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.info("Loading data...")
        images = torch.load(config["image_path"])
        logger.info(f"Images loaded. Shape: {images.shape}")

        catalog = torch.load(config["catalog_path"])
        logger.info("Catalogs loaded.")

        convert_save_images_catalogs(
            images,
            catalog,
            setting=config["setting"],
            shear_feature=config["shear_setting"],
        )
    except Exception as e:
        logger.info(f"Error occurred: {str(e)}")
        raise

    logger.info("=" * 60)
    logger.info(f"All processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {format_time(time.time() - start)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
