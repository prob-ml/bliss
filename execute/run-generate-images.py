#!/usr/bin/env python3
"""
Runs one of the other python files in execute in a loop or so. Nothing major should be in here.
Please run from outside the execute folder. In the root 'galaxy-net' directory.
"""
import subprocess
from os.path import dirname
from pathlib import Path

project_path = Path(dirname(dirname(__file__)))  # path to galaxy-net
path = f"PYTHONPATH={project_path.as_posix()} "


def main():
    # spawn multi-processes to generate images.
    ps = []
    for i in range(15):
        ps.append(
            subprocess.Popen(f"{path}"
                             f"./src/generate-images.py --dir test2 --filename images{i}"
                             f"--num-images 2000 --overwrite",
                             shell=True))

    # wait for all processes to complete.
    exit_codes = [p.communicate() for p in ps]

    # then we merge the results.
    subprocess.run(f"{path}"
                   f"./src/generate-images.py --dir test2 --merge",
                   shell=True)


if __name__ == '__main__':
    main()
