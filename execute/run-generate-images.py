#!/usr/bin/env python3
"""
Runs one of the other python files in execute in a loop or so. Nothing major should be in here.
Please run from outside the execute folder. In the root 'galaxy-net' directory.
"""
import subprocess
from __init__ import root_path


def main():
    # spawn multi-processes to generate images.
    ps = []
    for i in range(2):
        ps.append(
            subprocess.Popen(f"cd {root_path};"
                             f"python -m gmodel.generate_images --dir test1 --filename images{i} "
                             f"--num-images 20 --overwrite --slen 50 --generate",
                             shell=True))

    # wait for all processes to complete.
    _ = [p.communicate() for p in ps]

    # then we merge the results.
    subprocess.run(f"cd {root_path};"
                   f"python -m gmodel.generate_images --dir test1 --merge",
                   shell=True)


if __name__ == '__main__':
    main()
