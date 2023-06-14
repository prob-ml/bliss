from pathlib import Path

from bliss.utils.download_utils import download_git_lfs_file


def download_decals_base(download_dir: str):
    cutout_filename = "cutout_336.635_-0.9600.fits"
    tractor_filename = "tractor-3366m010.fits"
    cutout = download_git_lfs_file(
        f"https://api.github.com/repos/prob-ml/bliss/contents/data/decals/{cutout_filename}"
    )
    tractor = download_git_lfs_file(
        f"https://api.github.com/repos/prob-ml/bliss/contents/data/decals/{tractor_filename}"
    )
    cutout_path = Path(download_dir) / cutout_filename
    tractor_path = Path(download_dir) / tractor_filename
    cutout_path.write_bytes(cutout)
    tractor_path.write_bytes(tractor)
