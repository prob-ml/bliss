import logging
import os
import time
import traceback
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import glob

import torch
import yaml

# Import TileCatalog to match second script's approach
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


def get_device():
    """Get the best available device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    return device


def get_memory_usage():
    """Get current memory usage in GB."""
    memory = psutil.virtual_memory()
    used_gb = memory.used / 1024**3
    total_gb = memory.total / 1024**3
    percent = memory.percent
    return used_gb, total_gb, percent


def compress_shear(shear_tensor):
    """
    FIXED: Compress shear tensor without averaging (preserves original range)
    
    Expected input from TileCatalog: (n_tiles_per_side, n_tiles_per_side, max_num_of_sources, 1)
    Output: (n_tiles_per_side, n_tiles_per_side, 1)
    
    Since all sources in the same image should have identical shear values,
    we take the first valid shear value from each tile instead of averaging.
    """
    if len(shear_tensor.shape) == 4:
        # Input: (n_tiles, n_tiles, max_sources, 1) -> Output: (n_tiles, n_tiles, 1)
        # FIXED: Take first source's value instead of averaging
        return shear_tensor[:, :, 0, :]  # Shape: (n_tiles, n_tiles, 1)
        
    elif len(shear_tensor.shape) == 3:
        # Handle case where last dimension might be squeezed
        return shear_tensor[:, :, 0:1]
    else:
        logger.warning(f"Unexpected shear tensor shape: {shear_tensor.shape}")
        logger.warning("Expected 4D tensor (n_tiles, n_tiles, max_sources, 1)")
        # Fallback - return first element
        return shear_tensor.flatten()[0:1].view(1, 1, 1)


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    hours = seconds / 3600
    return f"{hours:.1f}h"


def find_batch_files(batch_dir, pattern="*"):
    """
    Find all batch files in a directory.
    
    Args:
        batch_dir: Directory containing batch files
        pattern: Pattern to match files (e.g., "batch_*.pt", "*.pt")
    
    Returns:
        List of sorted file paths
    """
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
    
    files = list(batch_path.glob(pattern))
    files.sort()  # Sort to ensure consistent processing order
    
    logger.info(f"Found {len(files)} batch files in {batch_dir}")
    for i, file in enumerate(files[:5]):  # Show first 5 files
        logger.info(f"  {i+1}. {file.name}")
    if len(files) > 5:
        logger.info(f"  ... and {len(files) - 5} more files")
    
    return files


def pair_image_catalog_batches(image_dir, catalog_dir, image_pattern="*", catalog_pattern="*"):
    """
    Find and pair image and catalog batch files.
    
    Args:
        image_dir: Directory containing image batch files
        catalog_dir: Directory containing catalog batch files
        image_pattern: Pattern for image files
        catalog_pattern: Pattern for catalog files
    
    Returns:
        List of (image_path, catalog_path) tuples
    """
    image_files = find_batch_files(image_dir, image_pattern)
    catalog_files = find_batch_files(catalog_dir, catalog_pattern)
    
    if len(image_files) != len(catalog_files):
        logger.warning(f"Mismatch in number of files: {len(image_files)} images vs {len(catalog_files)} catalogs")
    
    # Pair files by name or index
    paired_files = []
    for i, (img_file, cat_file) in enumerate(zip(image_files, catalog_files)):
        paired_files.append((img_file, cat_file))
        logger.info(f"Batch {i+1}: {img_file.name} + {cat_file.name}")
    
    return paired_files


class MemoryControlledBatchProcessor:
    """Memory-controlled batch processor that prevents OOM."""
    
    def __init__(self, max_memory_gb=100):
        self.max_memory_gb = max_memory_gb
        self.device = get_device()
        self.global_image_counter = 0  # Track cumulative image count across batches
        
    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    def process_single_batch(self, image_path, catalog_path, batch_id, save_folder):
        """
        Process a single batch of images and catalogs.
        
        Args:
            image_path: Path to image batch file
            catalog_path: Path to catalog batch file
            batch_id: Batch identifier for logging
            save_folder: Folder to save processed results
        
        Returns:
            Number of successfully processed images
        """
        logger.info(f"Processing batch {batch_id}...")
        logger.info(f"  Image file: {image_path}")
        logger.info(f"  Catalog file: {catalog_path}")
        
        batch_start_time = time.time()
        successful_saves = 0
        
        try:
            # Check memory before loading
            used_memory, total_memory, percent = get_memory_usage()
            logger.info(f"Memory before loading batch: {used_memory:.1f}/{total_memory:.1f} GB ({percent:.1f}%)")
            
            # Load images
            logger.info("Loading images...")
            images = torch.load(image_path, map_location="cpu")
            logger.info(f"Loaded {len(images)} images")
            
            # Load catalogs and create TileCatalog
            logger.info("Loading catalogs...")
            raw_catalogs = torch.load(catalog_path, map_location="cpu")
            tile_catalog = TileCatalog(raw_catalogs)
            logger.info("TileCatalog created")
            
            # Check memory after loading
            used_memory, total_memory, percent = get_memory_usage()
            logger.info(f"Memory after loading: {used_memory:.1f}/{total_memory:.1f} GB ({percent:.1f}%)")
            
            # Process each image in the batch
            num_images = len(images)
            for idx in tqdm(range(num_images), desc=f"Batch {batch_id}", unit="img"):
                try:
                    # Get image and shear data
                    image = images[idx]
                    shear_1 = tile_catalog["shear_1"][idx]
                    shear_2 = tile_catalog["shear_2"][idx]
                    
                    # Move to GPU for processing (if available)
                    if self.device.type == "cuda":
                        image_gpu = image.to(self.device)
                        shear_1_gpu = shear_1.to(self.device)
                        shear_2_gpu = shear_2.to(self.device)
                    else:
                        image_gpu = image
                        shear_1_gpu = shear_1
                        shear_2_gpu = shear_2
                    
                    # Compress shear tensors
                    compressed_shear_1 = compress_shear(shear_1_gpu)
                    compressed_shear_2 = compress_shear(shear_2_gpu)
                    
                    # Log compression results for first image overall (not just first batch)
                    if self.global_image_counter == 0:
                        logger.info(f"Shear compression results for first image:")
                        logger.info(f"  Input shear_1 shape: {shear_1.shape}")
                        logger.info(f"  Output shear_1 shape: {compressed_shear_1.shape}")
                        logger.info(f"  Input shear_2 shape: {shear_2.shape}")
                        logger.info(f"  Output shear_2 shape: {compressed_shear_2.shape}")
                    
                    # Create tile catalog
                    tile_catalog_data = {
                        "shear_1": compressed_shear_1.cpu(),
                        "shear_2": compressed_shear_2.cpu(),
                    }
                    
                    # Create dict data
                    dict_data = {
                        "images": image_gpu.cpu().clone(),
                        "tile_catalog": tile_catalog_data,
                    }
                    
                    # Save to disk with cumulative numbering
                    save_path = f"{save_folder}/dataset_{self.global_image_counter}_size_1.pt"
                    torch.save([dict_data], save_path)
                    
                    # Verify file was saved
                    if os.path.exists(save_path):
                        successful_saves += 1
                        if self.global_image_counter == 0:
                            file_size = os.path.getsize(save_path)
                            logger.info(f"Successfully saved {save_path} ({file_size} bytes)")
                    else:
                        logger.error(f"File was not saved: {save_path}")
                    
                    # Increment global counter
                    self.global_image_counter += 1
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Memory check every 10 images
                    if (idx + 1) % 10 == 0:
                        used_memory, total_memory, percent = get_memory_usage()
                        if percent > 85:
                            logger.warning(f"High memory usage: {used_memory:.1f} GB ({percent:.1f}%) - forcing cleanup")
                            self.cleanup_memory()
                
                except Exception as e:
                    logger.error(f"Error processing image {idx} in batch {batch_id}: {str(e)}")
                    continue
            
            # Clean up batch data
            del images, raw_catalogs, tile_catalog
            self.cleanup_memory()
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch {batch_id} completed in {format_time(batch_time)}")
            logger.info(f"Successfully processed: {successful_saves}/{num_images} images")
            logger.info(f"Cumulative images processed: {self.global_image_counter}")
            
            return successful_saves
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return 0


def process_multiple_batches(
    image_dir,
    catalog_dir,
    setting="meta",
    shear_feature="varying",
    max_memory_gb=100,
    image_pattern="*.pt",
    catalog_pattern="*.pt",
    start_batch=1,
    end_batch=None,
):
    """
    Process multiple batches of images and catalogs.
    
    Args:
        image_dir: Directory containing image batch files
        catalog_dir: Directory containing catalog batch files
        setting: Setting name for output folder
        shear_feature: Shear feature name for output folder
        max_memory_gb: Memory limit in GB
        image_pattern: Pattern to match image files
        catalog_pattern: Pattern to match catalog files
        start_batch: Starting batch number (1-indexed)
        end_batch: Ending batch number (None for all batches)
    """
    logger.info(f"Starting batch processing for {setting}_{shear_feature}_shear...")
    logger.info(f"Memory limit: {max_memory_gb} GB")
    
    overall_start_time = time.time()
    
    # Find and pair batch files
    try:
        batch_pairs = pair_image_catalog_batches(image_dir, catalog_dir, image_pattern, catalog_pattern)
    except Exception as e:
        logger.error(f"Error finding batch files: {str(e)}")
        return
    
    if not batch_pairs:
        logger.error("No batch files found!")
        return
    
    # Apply batch range filtering
    if end_batch is not None:
        batch_pairs = batch_pairs[start_batch-1:end_batch]
    else:
        batch_pairs = batch_pairs[start_batch-1:]
    
    logger.info(f"Will process {len(batch_pairs)} batches (from batch {start_batch})")
    
    # Create output folder
    save_folder = (
        f"/scratch/regier_root/regier0/taodingr/data/weak_lensing/"
        f"weak_lensing_img2048_{setting}_{shear_feature}_shear"
    )
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"Output folder: {save_folder}")
    
    # Initialize processor
    processor = MemoryControlledBatchProcessor(max_memory_gb)
    
    # Process each batch
    total_successful = 0
    total_processed = 0
    
    for batch_idx, (image_path, catalog_path) in enumerate(batch_pairs, start=start_batch):
        logger.info("=" * 50)
        logger.info(f"Processing batch {batch_idx}/{start_batch + len(batch_pairs) - 1}")
        logger.info("=" * 50)
        
        # Process single batch
        successful_saves = processor.process_single_batch(
            image_path, catalog_path, batch_idx, save_folder
        )
        
        total_successful += successful_saves
        total_processed += 1
        
        # Log progress
        elapsed_time = time.time() - overall_start_time
        batches_remaining = len(batch_pairs) - (batch_idx - start_batch + 1)
        if batch_idx > start_batch:
            avg_time_per_batch = elapsed_time / (batch_idx - start_batch + 1)
            eta = datetime.now() + timedelta(seconds=avg_time_per_batch * batches_remaining)
            logger.info(f"Batch progress: {batch_idx - start_batch + 1}/{len(batch_pairs)} | "
                       f"ETA: {eta.strftime('%H:%M:%S')}")
        
        # Memory status
        used_memory, total_memory, percent = get_memory_usage()
        logger.info(f"Memory after batch: {used_memory:.1f}/{total_memory:.1f} GB ({percent:.1f}%)")
        
        # Force cleanup between batches
        processor.cleanup_memory()
    
    # Final summary
    total_time = time.time() - overall_start_time
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETE!")
    logger.info(f"Processed batches: {total_processed}")
    logger.info(f"Total images processed: {processor.global_image_counter}")
    logger.info(f"Total successful saves: {total_successful}")
    logger.info(f"Total time: {format_time(total_time)}")
    logger.info(f"Average time per batch: {format_time(total_time / total_processed) if total_processed > 0 else 'N/A'}")
    logger.info(f"Output files: dataset_0_size_1.pt to dataset_{processor.global_image_counter-1}_size_1.pt")
    logger.info(f"Output folder: {save_folder}")
    logger.info("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info(
        f"Starting batch weak lensing data processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )
    logger.info("=" * 60)

    start = time.time()

    try:
        with open("encoder_input_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.info("Configuration:")
        logger.info(f"  Image directory: {config.get('image_dir', 'NOT SPECIFIED')}")
        logger.info(f"  Catalog directory: {config.get('catalog_dir', 'NOT SPECIFIED')}")
        logger.info(f"  Setting: {config['setting']}")
        logger.info(f"  Shear setting: {config['shear_setting']}")
        
        # Memory limit setting
        max_memory_gb = config.get("max_memory_gb", 100)
        logger.info(f"  Memory limit: {max_memory_gb} GB")
        
        # Batch processing settings
        image_pattern = config.get("image_pattern", "*.pt")
        catalog_pattern = config.get("catalog_pattern", "*.pt")
        start_batch = config.get("start_batch", 1)
        end_batch = config.get("end_batch", None)
        
        logger.info(f"  Image pattern: {image_pattern}")
        logger.info(f"  Catalog pattern: {catalog_pattern}")
        logger.info(f"  Batch range: {start_batch} to {end_batch if end_batch else 'end'}")
        
        # Check if directories exist
        image_dir = config.get('image_dir')
        catalog_dir = config.get('catalog_dir')
        
        if not image_dir:
            logger.error("image_dir not specified in config")
            return
        if not catalog_dir:
            logger.error("catalog_dir not specified in config")
            return
            
        if not os.path.exists(image_dir):
            logger.error(f"Image directory does not exist: {image_dir}")
            return
        if not os.path.exists(catalog_dir):
            logger.error(f"Catalog directory does not exist: {catalog_dir}")
            return

        logger.info("Starting batch processing...")
        process_multiple_batches(
            image_dir=image_dir,
            catalog_dir=catalog_dir,
            setting=config["setting"],
            shear_feature=config["shear_setting"],
            max_memory_gb=max_memory_gb,
            image_pattern=image_pattern,
            catalog_pattern=catalog_pattern,
            start_batch=start_batch,
            end_batch=end_batch,
        )
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

    logger.info("=" * 60)
    logger.info(f"All batch processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {format_time(time.time() - start)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()