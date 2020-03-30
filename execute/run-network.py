#!/usr/bin/env python3
"""
Runs one of the other python files in execute in a loop or so. Nothing major should be in here.
Please run from outside the execute folder. In the root 'galaxy-net' directory.
"""
import subprocess
import sys

from __init__ import root_path


def main(extra):
    cmd = (f"cd {root_path};"
           f"python -m gmodel.train_model --model centered_galaxy --dataset h5_catalog "
           f"--num-bands 6 --h5-file test1/images.hdf5 --epochs 200 --num-workers 0 --overwrite --evaluate 10 "
           f"--dir-name test1 --slen 50 "
           f"{extra}"  # optionally add one argument like -h
           )

    print(cmd)
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    s = ''
    if len(sys.argv) > 1:
        s = sys.argv[1]
    main(s)
