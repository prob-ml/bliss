import shutil
from pathlib import Path

from astropy.utils.data import download_file


def download_file_to_dst(url, dst_filename, preprocess_fn=lambda x: x):  # noqa: WPS404
    dst_path = Path(dst_filename)
    if dst_path.exists():
        return

    filename = download_file(url, cache=False, show_progress=False, timeout=120)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(filename, dst_filename)
    with open(dst_filename, "rb") as f:
        file_contents = f.read()
    file_contents = preprocess_fn(file_contents)
    with open(dst_filename, "wb") as f:
        f.write(file_contents)
