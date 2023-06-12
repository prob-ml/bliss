from pathlib import Path
from typing import Tuple

import torch
from astropy.table import Table
from astropy.wcs import WCS

from bliss.catalog import FullCatalog
from bliss.utils import column_to_tensor, convert_flux_to_mag


class DecalsFullCatalog(FullCatalog):
    """Class for the Decals Sweep Tractor Catalog.

    Some resources:
    - https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/
    - https://www.legacysurvey.org/dr9/files/#sweep-catalogs-region-sweep
    - https://www.legacysurvey.org/dr5/description/#photometry
    - https://www.legacysurvey.org/dr9/bitmasks/
    """

    @classmethod
    def from_file(
        cls,
        decals_cat_path: str,
        ra_lim: Tuple[int, int] = (-360, 360),
        dec_lim: Tuple[int, int] = (-90, 90),
        band: str = "r",
    ):
        """Loads DECaLS catalog from FITS file.

        Args:
            decals_cat_path (str): Path to .fits file.
            ra_lim (Tuple[int, int]): Range of RA values to keep.
                Defaults to (-360, 360).
            dec_lim (Tuple[int, int]): Range of DEC values to keep.
                Defaults to (-90, 90).
            band (str): Band to read from. Defaults to "r".

        Returns:
            A DecalsFullCatalog containing data from the provided file. Note that the
            coordinates in (RA, DEC) are not converted to plocs by default. For this,
            use get_plocs_from_ra_dec after loading the data.
        """
        catalog_path = Path(decals_cat_path)
        table = Table.read(catalog_path, format="fits", unit_parse_strict="silent")
        table = {k.upper(): v for k, v in table.items()}  # uppercase keys
        band = band.capitalize()

        # filter out pixels that aren't in primary region, had issues with source fitting,
        # in SGA large galaxy, or in a globular cluster. In the future this should probably
        # be an input parameter.
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

        # filter on locations
        # first lims imposed on frame
        keep_coord = (ra > ra_lim[0]) & (ra < ra_lim[1]) & (dec > dec_lim[0]) & (dec < dec_lim[1])
        keep = (galaxy_bool | star_bool) & keep_coord

        # filter quantities
        objid = objid[keep]
        galaxy_bool = galaxy_bool[keep]
        star_bool = star_bool[keep]
        ra = ra[keep]
        dec = dec[keep]
        flux = flux[keep]
        mag = convert_flux_to_mag(flux, 1)
        nobj = objid.shape[0]

        d = {
            "objid": objid.reshape(1, nobj, 1),
            "ra": ra.reshape(1, nobj, 1),
            "dec": dec.reshape(1, nobj, 1),
            "plocs": torch.zeros((1, nobj, 2)),  # compatibility with FullCatalog
            "n_sources": torch.tensor((nobj,)),
            "galaxy_bools": galaxy_bool.reshape(1, nobj, 1).float(),
            "star_bools": star_bool.reshape(1, nobj, 1).float(),
            "fluxes": flux.reshape(1, nobj, 1),
            "mags": mag.reshape(1, nobj, 1),
        }

        height = dec_lim[1] - dec_lim[0]
        width = ra_lim[1] - ra_lim[0]
        return cls(height, width, d)

    def get_plocs_from_ra_dec(self, wcs: WCS):
        """Converts RA/DEC coordinates into pixel coordinates.

        Args:
            wcs (WCS): WCS object to use for transformation.

        Returns:
            A 1xNx2 tensor containing the locations of the light sources in pixel coordinates. This
            function does not write self.plocs, so you should do that manually if necessary.
        """
        ra = self["ra"].numpy().squeeze()
        dec = self["dec"].numpy().squeeze()

        pt, pr = wcs.all_world2pix(ra, dec, 0)  # convert to pixel coordinates
        pt = torch.tensor(pt)
        pr = torch.tensor(pr)
        plocs = torch.stack((pr, pt), dim=-1)
        return plocs.reshape(1, plocs.size()[0], 2) + 0.5  # BLISS consistency
