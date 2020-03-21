#!/usr/bin/env python3
"""
Runs one of the other python files in execute in a loop or so. Nothing major should be in here.
Please run from outside the execute folder. In the root 'galaxy-net' directory.
"""
import subprocess
from __init__ import PYPATH


def main():
    # spawn multi-processes to generate images.
    ps = []
    for i in range(15):
        ps.append(
            subprocess.Popen(f"{PYPATH} "
                             f"./src/generate_images.py --dir test3 --filename images{i} "
                             f"--num-images 2000 --overwrite --slen 50 --generate",
                             shell=True))

    # wait for all processes to complete.
    _ = [p.communicate() for p in ps]

    # then we merge the results.
    subprocess.run(f"{PYPATH} "
                   f"./src/generate_images.py --dir test3 --merge",
                   shell=True)


if __name__ == '__main__':
    main()
