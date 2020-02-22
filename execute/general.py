#!/usr/bin/env python3
"""
Runs one of the other python files in execute in a loop or so. Nothing major should be in here.
Please run from outside the execute folder. In the root 'galaxy-net' directory.
"""
import subprocess
from os.path import dirname
from pathlib import Path

project_path = Path(dirname(dirname(__file__)))
path = f"PYTHONPATH={project_path.as_posix()}"

# for i in range(13):
#     subprocess.run(f"./execute/generate-images.py --dir test3 --filename images{i} --num-images 2000 --overwrite "
#                    f"--generate &",
#                    shell=True)

# subprocess.run(f"./execute/generate-images.py --dir test3 --merge", shell=True)

subprocess.run(f"{path} "
               f"./src/train_model.py -h",
               shell=True)
