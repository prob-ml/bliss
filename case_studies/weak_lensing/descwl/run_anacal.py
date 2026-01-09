#!/usr/bin/env python
"""AnaCal processing script for weak lensing simulations.

Runs AnaCal on the same test set used by NPE for fair comparison.

Usage:
    python run_anacal.py

Configure settings in config_run_anacal.yaml
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import anacal
import galsim
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from hydra import compose, initialize
from hydra.utils import instantiate
from tqdm import tqdm

from bliss.global_env import GlobalEnv

# Add bliss root to path
bliss_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if bliss_root not in sys.path:
    sys.path.insert(0, bliss_root)


# =============================================================================
# AnaCal Utility Classes
# =============================================================================


class GalSimPsfWrapper(anacal.psf.BasePsf):
    """Wrapper to make GalSim PSF objects compatible with anacal."""

    def __init__(self, galsim_psf, pixel_scale, npix=64, is_variable=False):
        self.galsim_psf = galsim_psf
        self.pixel_scale = pixel_scale
        self.npix = npix
        self.is_variable = is_variable
        self.shape = None

    def draw(self, x, y):
        """Draw PSF at position (x, y)."""
        if self.is_variable:
            pos = galsim.PositionD(x, y)
            psf_at_pos = self.galsim_psf.getPSF(pos)
        else:
            psf_at_pos = self.galsim_psf

        return psf_at_pos.drawImage(
            nx=self.npix, ny=self.npix, scale=self.pixel_scale, method="auto"
        ).array.astype(np.float64)


# =============================================================================
# Band Combination Functions
# =============================================================================


def combine_multiband_images(images_tensor, band_variances=None, method="inverse_variance"):
    """Combine multi-band images into single image for anacal processing."""
    if method == "inverse_variance":
        if band_variances is not None:
            weights = []
            band_names = ["g", "r", "i", "z"][: images_tensor.shape[0]]
            for band in band_names:
                variance_value = band_variances.get(band, 0.354)
                weights.append(1.0 / variance_value)
            weights = torch.tensor(weights, dtype=torch.float32)
        else:
            typical_variances = torch.tensor([0.099, 0.138, 0.354, 1.344])
            typical_variances = typical_variances[: images_tensor.shape[0]]
            weights = 1.0 / typical_variances

        total_weight = weights.sum()
        normalized_weights = weights / total_weight
        combined = torch.sum(images_tensor * normalized_weights.view(-1, 1, 1), dim=0)
        combined_variance = 1.0 / total_weight.item()

    elif method == "mean":
        combined = torch.mean(images_tensor, dim=0)
        combined_variance = 0.354

    else:
        raise ValueError(f"Unknown combination method: {method}")

    return combined, combined_variance


def combine_multiband_masks(masks_dict, method="union"):
    """Combine masks from multiple bands into a single mask."""
    if not masks_dict:
        return None

    band_names = list(masks_dict.keys())
    first_mask = masks_dict[band_names[0]]

    if method == "union":
        combined_mask = np.zeros_like(first_mask, dtype=np.int16)
        for mask in masks_dict.values():
            combined_mask = np.logical_or(combined_mask, mask).astype(np.int16)
    elif method == "first_only":
        combined_mask = first_mask.copy()
    elif method == "intersection":
        combined_mask = np.ones_like(first_mask, dtype=np.int16)
        for mask in masks_dict.values():
            combined_mask = np.logical_and(combined_mask, mask).astype(np.int16)
    else:
        raise ValueError(f"Unknown combination method: {method}")

    return combined_mask


# =============================================================================
# AnaCal Processing Functions
# =============================================================================


def anacal_multiband_combined(
    images_tensor,
    psf_input,
    masks_dict=None,
    combine_method="inverse_variance",
    mask_combine_method="union",
    star_catalog=None,
    npix=64,
    sigma_arcsec=0.52,
    mag_zero=30.0,
    pixel_scale=0.2,
    band_variances=None,
):
    """Process multi-band images by combining them first, then running anacal."""
    combined_image, combined_variance = combine_multiband_images(
        images_tensor, band_variances=band_variances, method=combine_method
    )
    gal_array = combined_image.numpy()

    combined_mask = None
    if masks_dict is not None:
        combined_mask = combine_multiband_masks(masks_dict, method=mask_combine_method)

    noise_array = np.random.normal(0, np.sqrt(combined_variance), gal_array.shape).astype(
        np.float64
    )

    fpfs_config = anacal.fpfs.FpfsConfig(npix=npix, sigma_arcsec=sigma_arcsec)

    if isinstance(psf_input, np.ndarray):
        out = anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=psf_input,
            pixel_scale=pixel_scale,
            noise_variance=combined_variance,
            noise_array=noise_array,
            mask_array=combined_mask,
            star_catalog=star_catalog,
            detection=None,
        )
    else:
        center_psf = psf_input.draw(gal_array.shape[1] // 2, gal_array.shape[0] // 2)
        out = anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=center_psf,
            psf_object=psf_input,
            pixel_scale=pixel_scale,
            noise_variance=combined_variance,
            noise_array=noise_array,
            mask_array=combined_mask,
            star_catalog=star_catalog,
            detection=None,
        )

    e1 = out["fpfs_w"] * out["fpfs_e1"]
    e1g1 = out["fpfs_dw_dg1"] * out["fpfs_e1"] + out["fpfs_w"] * out["fpfs_de1_dg1"]
    e2 = out["fpfs_w"] * out["fpfs_e2"]
    e2g2 = out["fpfs_dw_dg2"] * out["fpfs_e2"] + out["fpfs_w"] * out["fpfs_de2_dg2"]

    return np.sum(e1), np.sum(e1g1), np.sum(e2), np.sum(e2g2), len(e1)


# =============================================================================
# Processing Functions
# =============================================================================


def process_sample(sample, config):
    """Process a single sample with AnaCal."""
    images = sample["images"]
    catalog = sample["tile_catalog"]
    anacal_data = sample["anacal_data"]

    psf_image = anacal_data["psf_image"]
    masks_dict = anacal_data["masks"]
    band_variances = anacal_data["variances"]
    bright_star_catalog = anacal_data.get("bright_star_catalog")
    pixel_scale = anacal_data.get("pixel_scale", config["pixel_scale"])

    e1_sum, e1g1_sum, e2_sum, e2g2_sum, num_detections = anacal_multiband_combined(
        images,
        psf_image,
        masks_dict=masks_dict,
        combine_method=config["combine_method"],
        mask_combine_method=config["mask_combine_method"],
        star_catalog=bright_star_catalog,
        npix=config["npix"],
        sigma_arcsec=config["sigma_arcsec"],
        mag_zero=config["mag_zero"],
        pixel_scale=pixel_scale,
        band_variances=band_variances,
    )

    return {
        "e1_sum": float(e1_sum),
        "e1g1_sum": float(e1g1_sum),
        "e2_sum": float(e2_sum),
        "e2g2_sum": float(e2g2_sum),
        "num_detections": int(num_detections),
        "shear_1": float(catalog["shear_1"]),
        "shear_2": float(catalog["shear_2"]),
    }


def process_file(args):
    """Worker function for multiprocessing."""
    file_path, config = args
    try:
        data_list = torch.load(file_path, weights_only=False)
        results = []
        for sample in data_list:
            results.append(process_sample(sample, config))
        return results, None
    except Exception as e:
        return None, str(e)


def load_config():
    """Load anacal config from YAML file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_run_anacal.yaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("=== ANACAL PROCESSING ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load anacal-specific config
    cfg = load_config()
    print(f"Output: {cfg['output_file']}")

    # Use NPE's hydra config for data source
    with initialize(config_path="./", version_base=None):
        hydra_cfg = compose("config_train_npe")

    # Same seed as NPE for identical test set
    seed = pl.seed_everything(hydra_cfg.train.seed)
    GlobalEnv.seed_in_this_program = seed
    print(f"Seed: {hydra_cfg.train.seed}")

    # Get test files from NPE's data source
    print("Setting up data source...")
    data_source = instantiate(hydra_cfg.train.data_source)
    data_source.setup("test")
    test_files = data_source.test_dataset.file_paths
    print(f"Processing {len(test_files)} test files...")

    # Initialize results
    results = {
        "e1_sum": [],
        "e1g1_sum": [],
        "e2_sum": [],
        "e2g2_sum": [],
        "num_detections": [],
        "shear_1": [],
        "shear_2": [],
    }

    start_time = time.time()
    failed_count = 0
    n_workers = cfg.get("n_workers", 1)

    def append_results(file_results):
        for r in file_results:
            results["e1_sum"].append(r["e1_sum"])
            results["e1g1_sum"].append(r["e1g1_sum"])
            results["e2_sum"].append(r["e2_sum"])
            results["e2g2_sum"].append(r["e2g2_sum"])
            results["num_detections"].append(r["num_detections"])
            results["shear_1"].append(r["shear_1"])
            results["shear_2"].append(r["shear_2"])

    if n_workers > 1:
        print(f"Using {n_workers} workers...")
        args_list = [(f, cfg) for f in test_files]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_file, args): args[0] for args in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                file_results, error = future.result()
                if error:
                    print(f"Error: {error}")
                    failed_count += 1
                else:
                    append_results(file_results)
    else:
        for file_path in tqdm(test_files, desc="Processing"):
            file_results, error = process_file((file_path, cfg))
            if error:
                print(f"Error processing {file_path}: {error}")
                failed_count += 1
            else:
                append_results(file_results)

    # Save results
    elapsed_time = time.time() - start_time
    output = {
        "e1_sum": np.array(results["e1_sum"]),
        "e1g1_sum": np.array(results["e1g1_sum"]),
        "e2_sum": np.array(results["e2_sum"]),
        "e2g2_sum": np.array(results["e2g2_sum"]),
        "num_detections": np.array(results["num_detections"]),
        "shear_1": np.array(results["shear_1"]),
        "shear_2": np.array(results["shear_2"]),
        "total_e1_sum": float(np.sum(results["e1_sum"])),
        "total_e1g1_sum": float(np.sum(results["e1g1_sum"])),
        "total_e2_sum": float(np.sum(results["e2_sum"])),
        "total_e2g2_sum": float(np.sum(results["e2g2_sum"])),
        "total_detections": int(np.sum(results["num_detections"])),
        "n_samples": len(results["e1_sum"]),
        "processing_time": elapsed_time,
        "seed": hydra_cfg.train.seed,
    }

    torch.save(output, cfg["output_file"])
    print(f"\nSaved to: {cfg['output_file']}")

    # Summary
    print("\n=== COMPLETE ===")
    print(f"Samples processed: {output['n_samples']}")
    if failed_count > 0:
        print(f"Failed files: {failed_count}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Total detections: {output['total_detections']}")


if __name__ == "__main__":
    main()
