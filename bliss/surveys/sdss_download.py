import bz2
import gzip
from pathlib import Path

import requests

URLBASE = "https://data.sdss.org/sas/dr12/boss"
BANDS = ("u", "g", "r", "i", "z")


def download_file(url, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=10)
    with open(filename, "wb") as out_file:
        out_file.write(response.content)


def download_gz_file(url, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=10)
    with open(filename, "wb") as out_file:
        out_file.write(gzip.decompress(response.content))


def download_bz2_file(url, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=10)
    with open(filename, "wb") as out_file:
        out_file.write(bz2.decompress(response.content))


def download_all(run=94, camcol=1, field=12, download_dir="."):
    if not Path(download_dir).exists():
        # create download directory
        Path(download_dir).mkdir(parents=True, exist_ok=True)

    # strip leading zeros
    run_stripped = str(run).lstrip("0")
    field_stripped = str(field).lstrip("0")

    run6 = f"{int(run_stripped):06d}"
    field4 = f"{int(field_stripped):04d}"

    subdir2 = f"{run_stripped}/{camcol}"
    subdir3 = f"{run_stripped}/{camcol}/{field_stripped}"

    download_file(
        f"{URLBASE}/photoObj/301/{run_stripped}/photoField-{run6}-{camcol}.fits",
        f"{download_dir}/{subdir2}/photoField-{run6}-{camcol}.fits",
    )
    download_file(
        f"{URLBASE}/photoObj/301/{run_stripped}/{camcol}/photoObj-{run6}-{camcol}-{field4}.fits",
        f"{download_dir}/{subdir3}/photoObj-{run6}-{camcol}-{field4}.fits",
    )

    for band in BANDS:
        download_gz_file(
            f"{URLBASE}/photo/redux/301/{run_stripped}/objcs/{camcol}/"
            f"fpM-{run6}-{band}{camcol}-{field4}.fit.gz",
            f"{download_dir}/{subdir3}/fpM-{run6}-{band}{camcol}-{field4}.fits",
        )

        download_bz2_file(
            f"{URLBASE}/photoObj/frames/301/{run_stripped}/{camcol}/"
            f"frame-{band}-{run6}-{camcol}-{field4}.fits.bz2",
            f"{download_dir}/{subdir3}/frame-{band}-{run6}-{camcol}-{field4}.fits",
        )

    download_file(
        f"{URLBASE}/photo/redux/301/{run_stripped}/objcs/{camcol}/"
        f"psField-{run6}-{camcol}-{field4}.fit",
        f"{download_dir}/{subdir3}/psField-{run6}-{camcol}-{field4}.fits",
    )
