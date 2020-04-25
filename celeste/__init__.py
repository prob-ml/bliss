from os.path import dirname
import sys

src_path = dirname(__file__)
DeblendingStarfields_path = dirname(src_path)
packages_path = dirname(DeblendingStarfields_path)

paths = [src_path, DeblendingStarfields_path, packages_path]

if packages_path not in sys.path:  # avoid messing up sys.path if external import.
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
