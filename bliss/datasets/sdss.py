import pathlib
import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import Dataset
from astropy.io import fits
from astropy.wcs import WCS


class SloanDigitalSkySurvey(Dataset):
    def __init__(self, sdss_dir, run=3900, camcol=6, field=None, bands=[2]):
        super(SloanDigitalSkySurvey, self).__init__()

        self.sdss_path = pathlib.Path(sdss_dir)
        self.rcfgcs = []
        self.bands = bands

        # meta data for the run + camcol
        pf_file = "photoField-{:06d}-{:d}.fits".format(run, camcol)
        camcol_path = self.sdss_path.joinpath(str(run), str(camcol))
        pf_path = camcol_path.joinpath(pf_file)
        self.pf_fits = fits.getdata(pf_path)

        fieldnums = self.pf_fits["FIELD"]
        fieldgains = self.pf_fits["GAIN"]

        # get desired field
        for i in range(len(fieldnums)):
            _field = fieldnums[i]
            gain = fieldgains[i]
            if (not field) or _field == field:
                # load the catalog distributed with SDSS
                po_file = "photoObj-{:06d}-{:d}-{:04d}.fits".format(run, camcol, field)
                po_path = camcol_path.joinpath(str(field), po_file)
                po_fits = fits.getdata(po_path)

                self.rcfgcs.append((run, camcol, field, gain, po_fits))

        self.items = [None] * len(self.rcfgcs)

    def __len__(self):
        return len(self.rcfgcs)

    def __getitem__(self, idx):
        if not self.items[idx]:
            self.items[idx] = self.get_from_disk(idx)
        return self.items[idx]

    def fetch_bright_stars(self, po_fits, img, wcs):
        is_star = po_fits["objc_type"] == 6
        is_bright = po_fits["psfflux"].sum(axis=1) > 100
        is_thing = po_fits["thing_id"] != -1
        is_target = is_star & is_bright & is_thing
        ras = po_fits["ra"][is_target]
        decs = po_fits["dec"][is_target]

        band = 2
        stamps = []
        for (ra, dec, f) in zip(ras, decs, po_fits["thing_id"][is_target]):
            # pt = "time" in pixel coordinates
            pt, pr = wcs.wcs_world2pix(ra, dec, 0)
            pt, pr = int(pt + 0.5), int(pr + 0.5)
            stamp = img[(pr - 2) : (pr + 3), (pt - 2) : (pt + 3)]
            stamps.append(stamp)

        return np.asarray(stamps)

    def get_from_disk(self, idx):
        run, camcol, field, gain, po_fits = self.rcfgcs[idx]

        camcol_dir = self.sdss_path.joinpath(str(run), str(camcol))
        field_dir = camcol_dir.joinpath(str(field))

        image_list = []
        background_list = []
        nelec_per_nmgy_list = []
        calibration_list = []
        gain_list = []
        wcs_list = []

        cache_path = field_dir.joinpath("cache.pkl")

        for b, bl in enumerate("ugriz"):
            if not (b in self.bands):
                continue

            frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format(
                bl, run, camcol, field
            )
            frame_path = str(field_dir.joinpath(frame_name))
            print("loading sdss image from", frame_path)
            frame = fits.open(frame_path)

            calibration = frame[1].data
            nelec_per_nmgy = gain[b] / calibration

            (sky_small,) = frame[2].data["ALLSKY"]
            (sky_x,) = frame[2].data["XINTERP"]
            (sky_y,) = frame[2].data["YINTERP"]

            small_rows = np.mgrid[0 : sky_small.shape[0]]
            small_cols = np.mgrid[0 : sky_small.shape[1]]
            sky_interp = RegularGridInterpolator(
                (small_rows, small_cols), sky_small, method="nearest"
            )

            sky_y = sky_y.clip(0, sky_small.shape[0] - 1)
            sky_x = sky_x.clip(0, sky_small.shape[1] - 1)
            large_points = np.stack(np.meshgrid(sky_y, sky_x)).transpose()
            large_sky = sky_interp(large_points)
            large_sky_nelec = large_sky * gain[b]

            pixels_ss_nmgy = frame[0].data
            pixels_ss_nelec = pixels_ss_nmgy * nelec_per_nmgy
            pixels_nelec = pixels_ss_nelec + large_sky_nelec

            image_list.append(pixels_nelec)
            background_list.append(large_sky_nelec)

            gain_list.append(gain[b])
            nelec_per_nmgy_list.append(nelec_per_nmgy)
            calibration_list.append(calibration)

            wcs = WCS(frame[0])
            wcs_list.append(wcs)

            frame.close()

        stamps = self.fetch_bright_stars(po_fits, image_list[2], wcs_list[2])

        ret = {
            "image": np.stack(image_list),
            "background": np.stack(background_list),
            "nelec_per_nmgy": np.stack(nelec_per_nmgy_list),
            "gain": np.stack(gain_list),
            "calibration": np.stack(calibration_list),
            "wcs": wcs_list,
            "bright_stars": stamps,
        }
        pickle.dump(ret, field_dir.joinpath("cache.pkl").open("wb+"))

        return ret

def _get_mgrid2(slen0, slen1):
    offset0 = (slen0 - 1) / 2
    offset1 = (slen1 - 1) / 2
    x, y = np.mgrid[-offset0 : (offset0 + 1), -offset1 : (offset1 + 1)]
    return torch.Tensor(np.dstack((y, x))) / torch.Tensor([[[offset1, offset0]]])

def convert_mag_to_nmgy(mag):
    return 10 ** ((22.5 - mag) / 2.5)


def convert_nmgy_to_mag(nmgy):
    return 22.5 - 2.5 * torch.log10(nmgy)


def load_m2_data(
    sdss_dir="../../sdss_stage_dir/",
    hubble_dir="../hubble_data/",
    slen=100,
    x0=630,
    x1=310,
    f_min=1000.0,
    border_padding=0,
):
    # returns the SDSS image of M2 in the r and i bands
    # along with the corresponding Hubble catalog

    #####################
    # Load SDSS data
    #####################
    run = 2583
    camcol = 2
    field = 136

    sdss_data = SloanDigitalSkySurvey(
        sdss_dir,
        run=run,
        camcol=camcol,
        field=field,
        # returns the r and i band
        bands=[2, 3],
    )

    # the full SDSS image, ~1500 x 2000 pixels
    sdss_image_full = torch.Tensor(sdss_data[0]["image"])
    sdss_background_full = torch.Tensor(sdss_data[0]["background"])

    #####################
    # load hubble catalog
    #####################
    hubble_cat_file = (
        hubble_dir + "hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt"
    )
    print("loading hubble data from ", hubble_cat_file)
    HTcat = np.loadtxt(hubble_cat_file, skiprows=True)

    # hubble magnitude
    hubble_rmag_full = HTcat[:, 9]
    # right ascension and declination
    hubble_ra_full = HTcat[:, 21]
    hubble_dc_full = HTcat[:, 22]

    # convert hubble r.a and declination to pixel coordinates
    # (0, 0) is top left of sdss_image_full
    frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format("r", run, camcol, field)
    field_dir = pathlib.Path(sdss_dir).joinpath(str(run), str(camcol), str(field))
    frame_path = str(field_dir.joinpath(frame_name))
    print("getting sdss coordinates from: ", frame_path)
    hdulist = fits.open(str(frame_path))
    wcs = WCS(hdulist["primary"].header)
    # NOTE: pix_coordinates are (column x row), i.e. pix_coord[0] corresponds to a column
    pix_coordinates = wcs.wcs_world2pix(
        hubble_ra_full, hubble_dc_full, 0, ra_dec_order=True
    )
    hubble_locs_full_x0 = pix_coordinates[1]  # the row of pixel
    hubble_locs_full_x1 = pix_coordinates[0]  # the column of pixel

    # convert hubble magnitude to n_electron count
    # only take r band
    nelec_per_nmgy_full = sdss_data[0]["nelec_per_nmgy"][0].squeeze()
    which_cols = np.floor(hubble_locs_full_x1 / len(nelec_per_nmgy_full)).astype(int)
    hubble_nmgy = convert_mag_to_nmgy(hubble_rmag_full)

    hubble_r_fluxes_full = hubble_nmgy * nelec_per_nmgy_full[which_cols]

    #####################
    # using hubble locations,
    # align i-band with r-band
    # TODO: we don't actually need hubble locations ... random locations will do
    # separate function to load data from function to load sdss ... 
    # so that it is absolutely clear we do not use hubble data in the analysis
    #####################
    frame_name_i = "frame-{}-{:06d}-{:d}-{:04d}.fits".format("i", run, camcol, field)
    frame_path_i = str(field_dir.joinpath(frame_name_i))
    print("\n aligning images. \n Getting sdss coordinates from: ", frame_path_i)
    hdu = fits.open(str(frame_path_i))
    wcs_other = WCS(hdu["primary"].header)

    # get pixel coords
    pix_coordinates_other = wcs_other.wcs_world2pix(
        hubble_ra_full, hubble_dc_full, 0, ra_dec_order=True
    )

    # estimate the amount to shift
    shift_x0 = np.median(hubble_locs_full_x0 - pix_coordinates_other[1]) / (
        sdss_image_full.shape[-2] - 1
    )
    shift_x1 = np.median(hubble_locs_full_x1 - pix_coordinates_other[0]) / (
        sdss_image_full.shape[-1] - 1
    )
    shift = torch.Tensor([[[[shift_x1, shift_x0]]]]) * 2

    # align image
    grid = (
        _get_mgrid2(sdss_image_full.shape[-2], sdss_image_full.shape[-1]).unsqueeze(0)
        - shift
    )
    sdss_image_full[1] = torch.nn.functional.grid_sample(
        sdss_image_full[1].unsqueeze(0).unsqueeze(0), grid, align_corners=True
    ).squeeze()
    ##################
    # Filter to desired subimage
    ##################
    print("\n returning image at x0 = {}, x1 = {}".format(x0, x1))
    which_locs = (
        (hubble_locs_full_x0 > x0)
        & (hubble_locs_full_x0 < (x0 + slen - 1))
        & (hubble_locs_full_x1 > x1)
        & (hubble_locs_full_x1 < (x1 + slen - 1))
    )

    # just a subset
    sdss_image = sdss_image_full[
        :,
        (x0 - border_padding) : (x0 + slen + border_padding),
        (x1 - border_padding) : (x1 + slen + border_padding),
    ].to(device)
    sdss_background = sdss_background_full[
        :,
        (x0 - border_padding) : (x0 + slen + border_padding),
        (x1 - border_padding) : (x1 + slen + border_padding),
    ].to(device)

    locs = np.array(
        [hubble_locs_full_x0[which_locs] - x0, hubble_locs_full_x1[which_locs] - x1]
    ).transpose()
    hubble_r_fluxes = torch.Tensor(hubble_r_fluxes_full[which_locs])

    # hubble_locs = torch.Tensor(locs) / (slen - 1)
    # different from my old repo ...
    hubble_locs = (torch.Tensor(locs) + 0.5) / slen
    hubble_fluxes = torch.stack([hubble_r_fluxes, hubble_r_fluxes]).transpose(0, 1)

    # filter by bright stars only
    which_bright = hubble_fluxes[:, 0] > f_min
    hubble_locs = hubble_locs[which_bright].to(device)
    hubble_fluxes = hubble_fluxes[which_bright].to(device)

    return sdss_image, sdss_background, hubble_locs, hubble_fluxes, sdss_data, wcs
