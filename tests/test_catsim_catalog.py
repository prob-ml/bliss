from pathlib import Path

import numpy as np
from astropy.table import Table


def test_galaxy_blend_catalogs(home_dir: Path):
    cat = Table.read(home_dir / "data" / "OneDegSq.fits")

    pa_bulge = cat["pa_bulge"]
    pa_disk = cat["pa_disk"]

    # check these are in degrees
    assert np.max(pa_bulge) > 350
    assert np.max(pa_bulge) <= 360
    assert np.min(pa_bulge) < 1
    assert np.min(pa_bulge) >= 0

    assert np.max(pa_disk) > 350
    assert np.max(pa_disk) <= 360
    assert np.min(pa_disk) < 1
    assert np.min(pa_disk) >= 0
