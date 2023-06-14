import bz2
import gzip
from pathlib import Path

import requests

__all__ = [  # noqa: WPS410
    "download_pf",
    "download_po",
    "download_frame",
    "download_psfield",
    "download_all",
]

URLBASE = "https://data.sdss.org/sas/dr12/boss"
BANDS = ("u", "g", "r", "i", "z")


def download_pf(run=94, camcol=1, download_dir="."):
    download_file(
        f"{URLBASE}/photoObj/301/{_stripped(run)}/photoField-{_run6(run)}-{camcol}.fits",
        f"{download_dir}/{_subdir2(run, camcol)}/photoField-{_run6(run)}-{camcol}.fits",
    )


def download_po(run=94, camcol=1, field=12, download_dir="."):
    download_file(
        f"{URLBASE}/photoObj/301/{_stripped(run)}/{camcol}/"
        f"photoObj-{_run6(run)}-{camcol}-{_field4(field)}.fits",
        f"{download_dir}/{_subdir3(run, camcol, field)}/"
        f"photoObj-{_run6(run)}-{camcol}-{_field4(field)}.fits",
    )


def download_frame(run=94, camcol=1, field=12, band="r", download_dir="."):
    download_file(
        f"{URLBASE}/photo/redux/301/{_stripped(run)}/objcs/{camcol}/"
        f"fpM-{_run6(run)}-{band}{camcol}-{_field4(field)}.fit.gz",
        f"{download_dir}/{_subdir3(run, camcol, field)}/"
        f"fpM-{_run6(run)}-{band}{camcol}-{_field4(field)}.fits",
        gzip.decompress,
    )

    download_file(
        f"{URLBASE}/photoObj/frames/301/{_stripped(run)}/{camcol}/"
        f"frame-{band}-{_run6(run)}-{camcol}-{_field4(field)}.fits.bz2",
        f"{download_dir}/{_subdir3(run, camcol, field)}/"
        f"frame-{band}-{_run6(run)}-{camcol}-{_field4(field)}.fits",
        bz2.decompress,
    )


def download_psfield(run=94, camcol=1, field=12, download_dir="."):
    download_file(
        f"{URLBASE}/photo/redux/301/{_stripped(run)}/objcs/{camcol}/"
        f"psField-{_run6(run)}-{camcol}-{_field4(field)}.fit",
        f"{download_dir}/{_subdir3(run, camcol, field)}/"
        f"psField-{_run6(run)}-{camcol}-{_field4(field)}.fits",
    )


def download_all(run=94, camcol=1, field=12, download_dir="."):
    if not Path(download_dir).exists():
        # create download directory
        Path(download_dir).mkdir(parents=True, exist_ok=True)

    download_pf(run, camcol, download_dir)
    download_po(run, camcol, field, download_dir)

    for band in BANDS:
        download_frame(run, camcol, field, band, download_dir)

    download_psfield(run, camcol, field, download_dir)


# Helper functions
def download_file(url, filename, preprocess_fn=lambda x: x):  # noqa: WPS404
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=10)
    with open(filename, "wb") as out_file:
        out_file.write(preprocess_fn(response.content))


def _stripped(val):
    return str(val).lstrip("0")


def _run6(run):
    return f"{int(_stripped(run)):06d}"


def _field4(field):
    return f"{int(_stripped(field)):04d}"


def _subdir2(run, camcol):
    return f"{_stripped(run)}/{camcol}"


def _subdir3(run, camcol, field):
    return f"{_stripped(run)}/{camcol}/{_stripped(field)}"
