#!/usr/bin/env python3
"""
Runs one of the other python files in execute in a loop or so. Nothing major...
"""
import subprocess

# for i in range(13):
#     subprocess.run(f"./execute/generate-images.py --dir test3 --filename images{i} --num-images 2000 --overwrite "
#                    f"--generate &",
#                    shell=True)

subprocess.run(f"./execute/generate-images.py --dir test3 --merge", shell=True)
