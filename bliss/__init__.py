from os import getenv, environ
from pathlib import Path

if getenv("BLISS_HOME") is None:
    bliss_path = Path(__file__).resolve()
    environ["BLISS_HOME"] = bliss_path.parent.parent.as_posix()
