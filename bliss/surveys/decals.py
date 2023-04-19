from pathlib import Path
from typing import Tuple

import torch
from astropy.table import Table
from astropy.wcs import WCS

from bliss.catalog import FullCatalog
from bliss.surveys.sdss import column_to_tensor


class DecalsFullCatalog(FullCatalog):
    """Class for the Decals Sweep Tractor Catalog.

    Some resources:
    - https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/
    - https://www.legacysurvey.org/dr9/files/#sweep-catalogs-region-sweep
    - https://www.legacysurvey.org/dr5/description/#photometry
    - https://www.legacysurvey.org/dr9/bitmasks/
    """

    @staticmethod
    def _flux_to_mag(flux):
        return 22.5 - 2.5 * torch.log10(flux)

    @classmethod
    def from_file(
        cls,
        decals_cat_path,
        wcs: WCS,
        hlim: Tuple[int, int],  # in degrees
        wlim: Tuple[int, int],  # in degrees
        band: str = "r",
        mag_max=23,
    ):
        assert hlim[0] == wlim[0]
        catalog_path = Path(decals_cat_path)
        table = Table.read(catalog_path, format="fits")
        band = band.capitalize()

        ra_lim, dec_lim = wcs.all_pix2world(wlim, hlim, 0)
        bitmask = 0b0011010000000001  # noqa: WPS339

        objid = column_to_tensor(table, "OBJID")
        objc_type = table["TYPE"].data.astype(str)
        bits = table["MASKBITS"].data.astype(int)
        is_galaxy = torch.from_numpy(
            (objc_type == "DEV")
            | (objc_type == "REX")
            | (objc_type == "EXP")
            | (objc_type == "SER")
        )
        is_star = torch.from_numpy(objc_type == "PSF")
        ra = column_to_tensor(table, "RA")
        dec = column_to_tensor(table, "DEC")
        flux = column_to_tensor(table, f"FLUX_{band}")
        mask = torch.from_numpy((bits & bitmask) == 0).bool()

        galaxy_bool = is_galaxy & mask & (flux > 0)
        star_bool = is_star & mask & (flux > 0)

        # get pixel coordinates
        pt, pr = wcs.all_world2pix(ra, dec, 0)  # pixels
        pt = torch.tensor(pt)
        pr = torch.tensor(pr)
        ploc = torch.stack((pr, pt), dim=-1)

        # filter on locations
        # first lims imposed on frame
        keep_coord = (ra > ra_lim[0]) & (ra < ra_lim[1]) & (dec > dec_lim[0]) & (dec < dec_lim[1])
        # then, SDSS saturation
        regions = [(1200, 1360, 1700, 1900), (280, 400, 1220, 1320)]
        keep_sat = torch.ones(len(ra)).bool()
        for lim in regions:
            keep_sat = keep_sat & ((pr < lim[0]) | (pr > lim[1]) | (pt < lim[2]) | (pt > lim[3]))
        keep = (galaxy_bool | star_bool) & keep_coord & keep_sat

        # filter quantities
        objid = objid[keep]
        galaxy_bool = galaxy_bool[keep]
        star_bool = star_bool[keep]
        ra = ra[keep]
        dec = dec[keep]
        flux = flux[keep]
        mag = cls._flux_to_mag(flux)
        ploc = ploc[keep, :]
        nobj = ploc.shape[0]

        d = {
            "objid": objid.reshape(1, nobj, 1),
            "ra": ra.reshape(1, nobj, 1),
            "dec": dec.reshape(1, nobj, 1),
            "plocs": ploc.reshape(1, nobj, 2) - hlim[0] + 0.5,  # BLISS consistency
            "n_sources": torch.tensor((nobj,)),
            "galaxy_bools": galaxy_bool.reshape(1, nobj, 1).float(),
            "star_bools": star_bool.reshape(1, nobj, 1).float(),
            "fluxes": flux.reshape(1, nobj, 1),
            "mags": mag.reshape(1, nobj, 1),
        }

        height = hlim[1] - hlim[0]
        width = wlim[1] - wlim[0]
        full_cat = cls(height, width, d)
        return full_cat.apply_param_bin("mags", 0, mag_max)
