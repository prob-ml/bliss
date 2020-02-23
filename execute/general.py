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


# for i in range(15):
#     subprocess.run(f"{path}"
#                    f"./src/generate-images.py --dir test2 --filename images{i} --num-images 2000 --overwrite "
#                    f"--generate &",
#                    shell=True)

# subprocess.run(f"{path}"
#                f"./src/generate-images.py --dir test2 --merge", shell=True)


def main():
    subprocess.run(f"{path} "
                   f"./src/train_model.py --model centered_galaxy --dataset h5_catalog --num-bands 6 "
                   f"--epochs 100 --num-workers 0 --overwrite --h5-file test2/images.hdf5 --evaluate 10 "
                   f"--dir-name test3",
                   shell=True)


if __name__ == '__main__':
    main()
