#!/usr/bin/env python3
"""
Runs one of the other python files in execute in a loop or so. Nothing major should be in here.
Please run from outside the execute folder. In the root 'galaxy-net' directory.
"""
import subprocess
from os.path import dirname
from pathlib import Path
import sys

project_path = Path(dirname(dirname(__file__)))  # path to galaxy-net
path = f"PYTHONPATH={project_path.as_posix()} "


def main():
    subprocess.run(f"{path} "
                   f"./src/train_model.py --model centered_galaxy --dataset h5_catalog"
                   f"--num-bands 6 --h5-file test2/images.hdf5 --epochs 100 --num-workers 0 --overwrite  --evaluate 10 "
                   f"--dir-name test3 "
                   f"{sys.argv[1]}",  # optionally add one argument like -h
                   shell=True)


if __name__ == '__main__':
    main()
