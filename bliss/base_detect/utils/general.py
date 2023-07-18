# flake8: noqa
import glob
import inspect
import logging
import math
import os
import platform
import sys
import urllib
from datetime import datetime
from logging import config
from pathlib import Path
from subprocess import check_output
from typing import Optional

import cv2
import numpy as np
import pkg_resources as pkg
import torch

from bliss.base_detect.utils import emojis

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv("RANK", -1))  # pylint: disable=W1508

# Settings
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
DATASETS_DIR = Path(
    os.getenv("YOLOv5_DATASETS_DIR", ROOT.parent / "datasets")
)  # global datasets directory
AUTOINSTALL = (
    str(os.getenv("YOLOv5_AUTOINSTALL", True)).lower() == "true"  # pylint: disable=W1508
)  # global auto-install mode
VERBOSE = str(os.getenv("YOLOv5_VERBOSE", True)).lower() == "true"  # pylint: disable=W1508
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
FONT = "Arial.ttf"  # https://ultralytics.com/assets/Arial.ttf

torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(
    linewidth=320, formatter={"float_kind": "{:11.5g}".format}
)  # format short g, %precision=5
# prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
cv2.setNumThreads(0)  # pylint: disable=E1101
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["OMP_NUM_THREADS"] = (
    "1" if platform.system() == "darwin" else str(NUM_THREADS)
)  # OpenMP (PyTorch and SciPy)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress verbose TF compiler warnings in Colab


LOGGING_NAME = "yolov5"


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    # rank in world for Multi-GPU trainings
    rank = int(os.getenv("RANK", -1))  # pylint: disable=W1508
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {
                name: {
                    "level": level,
                    "handlers": [name],
                    "propagate": False,
                }
            },
        }
    )


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(
    LOGGING_NAME
)  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == "Windows":
    for fn in LOGGER.info, LOGGER.warning:
        # emoji safe logging
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # pylint: disable=W0640


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(  # pylint: disable=W1201
        colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items())
    )


def check_version(
    current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False
):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed"  # pylint: disable=C0301
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


def file_date(path=__file__):
    # Return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def git_describe(path=ROOT):  # path must be a directory
    # Return human-readable git description,
    # i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    try:
        assert (Path(path) / ".git").is_dir()
        return check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[
            :-1
        ]
    except Exception:  # pylint: disable=W0718
        return ""


def check_yaml(file, suffix=(".yaml", ".yml")):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_file(file, suffix=""):
    # pylint: disable=W1203
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # pylint: disable=R1705
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(
            urllib.parse.unquote(file).split("?")[0]
        ).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            LOGGER.info(f"Found {url} locally at {file}")  # file already exists
        else:
            LOGGER.info(f"Downloading {url} to {file}...")
            torch.hub.download_url_to_file(url, file)
            assert (
                Path(file).exists() and Path(file).stat().st_size > 0
            ), f"File download failed: {url}"  # check
        return file
    elif file.startswith("clearml://"):  # ClearML Dataset ID
        assert (
            "clearml" in sys.modules
        ), "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."  # pylint: disable=C0301
        return file
    else:  # search
        files = []
        for d in "data", "models", "utils":  # search directories
            files.extend(glob.glob(str(ROOT / d / "**" / file), recursive=True))  # find file
        assert len(files), f"File not found: {file}"  # pylint: disable=C1802
        assert (
            len(files) == 1
        ), f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def colorstr(*input):
    # pylint: disable=W0622
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code,
    # i.e.  colorstr('blue', 'hello world')
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def clip_segments(segments, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y
