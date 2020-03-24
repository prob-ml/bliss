from os.path import dirname
import sys

src_path = dirname(__file__)
GalaxyModel_path = dirname(src_path)
packages_path = dirname(GalaxyModel_path)

paths = [src_path, GalaxyModel_path]

if packages_path not in sys.path:  # avoid messing up sys.path if external import.
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
