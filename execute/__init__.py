from os.path import dirname
from pathlib import Path

project_path = Path(dirname(dirname(__file__)))  # path to galaxy-net
PYPATH = f"PYTHONPATH={project_path.as_posix()} "  # constant.
