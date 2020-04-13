#!/usr/bin/env python3
"""
Runs one of the other python files in execute in a loop or so. Nothing major should be in here.
Please run from outside the execute folder. In the root 'galaxy-net' directory.
"""
import subprocess
from __init__ import root_path


def main():
    # spawn multi-processes to generate images.
    num_copies = 10
    num_images = 1000
    dir_name = "test2"
    num_bands = 1
    slen = 51
    ps = []

    for i in range(num_copies):
        ps.append(
            subprocess.Popen(f"cd {root_path};"
                             f"python -m gmodel.generate_images --dir {dir_name} --filename images{i} "
                             f"--num-images {num_images} --overwrite --slen {slen} --generate --num-bands {num_bands}",
                             shell=True))

    # wait for all processes to complete.
    _ = [p.communicate() for p in ps]

    # then we merge the results.
    subprocess.run(f"cd {root_path};"
                   f"python -m gmodel.generate_images --dir {dir_name} --merge",
                   shell=True)


if __name__ == '__main__':
    main()
