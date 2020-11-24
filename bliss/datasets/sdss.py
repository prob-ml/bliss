import pathlib
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import fitsio
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import Dataset
from astropy.io import fits
from astropy.wcs import WCS
from math import floor, ceil

def construct_subpixel_grid_base(size_x, size_y):
    grid_x = np.linspace(-1.0, 1.0, num=size_x)
    grid_y = np.linspace(-1.0, 1.0, num=size_y)

    G_x    = torch.stack([torch.from_numpy(grid_x)]*size_y, dim=0)
    G_y    = torch.stack([torch.from_numpy(grid_y)]*size_x, dim=1)

    G      = torch.stack([G_x, G_y], dim=2)

    return G

def construct_subpixel_grid(G, shift_x, shift_y):
    G_shift = G.clone()
    G_shift[:, :, 0] += shift_x
    G_shift[:, :, 1] += shift_y
    return G_shift

def center_stamp_subpixel(stamp, pt, pr, G):
    stamp_torch = torch.from_numpy(stamp)
    size_y, size_x = stamp.shape
    shift_x     = 2*(pt - int(pt + 0.5)) / size_x
    shift_y     = 2*(pr - int(pr + 0.5)) / size_y
    G_shift     = construct_subpixel_grid(G, shift_x, shift_y).float()
    stamp_shifted = F.grid_sample(stamp_torch.unsqueeze(0).unsqueeze(0), G_shift.unsqueeze(0), align_corners=False)
    return stamp_shifted.squeeze(0).squeeze(0).numpy()

def read_psf(psf_fit_file, band=2):
    psfield = fitsio.FITS(psf_fit_file)
    hdu     = psfield[band+1]
    psf     = hdu.read()
    return {'hdu': hdu, 'psf': psf}

class SdssPSF:
    def __init__(self, psf_fit_file, bands):
        self.psf_fit_file = psf_fit_file
        self.bands        = bands
        self.init_cache()

    def init_cache(self):
        self.cache        = [None] * len(self.bands)

    def __getitem__(self, idx):
        if self.cache[idx] is None:
            x = read_psf(self.psf_fit_file, band=self.bands[idx])
            self.cache[idx] = x
        return self.cache[idx]

def psf_at_points(x, y, psf_data):
    hdu = psf_data['hdu']
    psf = psf_data['psf']
    # psfield = fitsio.FITS(psf_fit_file)
    # hdu = psfield[3]

    rtnscalar = np.isscalar(x) and np.isscalar(y)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # psf = hdu.read()

    psfimgs = None
    (outh, outw) = (None, None)

    # From the IDL docs:
    # http://photo.astro.princeton.edu/photoop_doc.html#SDSS_PSF_RECON
    #   acoeff_k = SUM_i{ SUM_j{ (0.001*ROWC)^i * (0.001*COLC)^j * C_k_ij } }
    #   psfimage = SUM_k{ acoeff_k * RROWS_k }
    for k in range(len(psf)):
        nrb = psf[k]["nrow_b"]
        ncb = psf[k]["ncol_b"]

        c = psf[k]["c"].reshape(5, 5)
        c = c[:nrb, :ncb]

        (gridi, gridj) = np.meshgrid(range(nrb), range(ncb))

        if psfimgs is None:
            psfimgs = [np.zeros_like(hdu["rrows"].read()[k]) for xy in np.broadcast(x, y)]
            (outh, outw) = (hdu["rnrow"].read()[k], hdu["rncol"].read()[k])

        for i, (xi, yi) in enumerate(np.broadcast(x, y)):
            acoeff_k = (((0.001 * xi) ** gridi * (0.001 * yi) ** gridj * c)).sum()
            psfimgs[i] += acoeff_k * hdu["rrows"].read()[k]

    psfimgs = [img.reshape((outh, outw)) for img in psfimgs]

    if rtnscalar:
        return psfimgs[0]
    return psfimgs
class SloanDigitalSkySurvey(Dataset):
    def __init__(self, sdss_dir, run=3900, camcol=6, fields=None, bands=[2], stampsize=5, overwrite_fits_cache=False, overwrite_psf_cache=True, overwrite_cache=False, center_subpixel=True):
        super(SloanDigitalSkySurvey, self).__init__()

        self.sdss_path = pathlib.Path(sdss_dir)
        self.rcfgcs = []
        self.bands = bands
        self.stampsize=stampsize
        self.overwrite_cache = overwrite_cache
        self.center_subpixel = center_subpixel

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
            if (not fields) or _field in fields:
                # load the catalog distributed with SDSS
                po_file  = "photoObj-{:06d}-{:d}-{:04d}.fits".format(run, camcol, _field)
                po_cache = "photoObj-{:06d}-{:d}-{:04d}.pkl".format(run, camcol, _field)
                po_path  = camcol_path.joinpath(str(_field), po_file)
                po_cache_path = camcol_path.joinpath(str(_field), po_cache)
                if (not po_cache_path.exists()) or (overwrite_fits_cache):
                    try:
                        po_fits = fits.getdata(po_path)
                    except IndexError as e:
                            print("Warning: IndexError while accessing field: {}. This field will not be included.".format(_field))
                            print(e)
                            po_fits = None
                    pickle.dump(po_fits, po_cache_path.open("wb+"))
                else:
                    po_fits = pickle.load(po_cache_path.open("rb+"))
                    if po_fits is None:
                        print("Warning: cached data for field {} is None. This field will not be included".format(_field))

                # Load the PSF produced by SDSS
                psf_file  = "psField-{:06d}-{:d}-{:04d}.fits".format(run, camcol, _field)
                psf_cache = "psField-{:06d}-{:d}-{:04d}.pkl".format(run, camcol, _field)

                psf_path       = camcol_path.joinpath(str(_field), psf_file)
                psf_cache_path = camcol_path.joinpath(str(_field), psf_cache)
                if (not psf_cache_path.exists()) or (overwrite_psf_cache):
                    try:
                        psf = SdssPSF(psf_path.as_posix(), bands)
                    except IndexError as e:
                            print("Warning: IndexError while accessing field: {}. This field will not be included.".format(_field))
                            print(e)
                            psf = None
                    pickle.dump(psf, psf_cache_path.open("wb+"))
                else:
                    raise(Exception("cannot currently use cache with PSF"))
                    psf = pickle.load(psf_cache_path.open("rb+"))
                    if psf is None:
                        print("Warning: cached psf data for field {} is None.".format(_field))

                if po_fits is not None:
                    self.rcfgcs.append((run, camcol, _field, gain, po_fits, psf))
        self.items = [None] * len(self.rcfgcs)

    def __len__(self):
        return len(self.rcfgcs)

    def __getitem__(self, idx):
        if not self.items[idx]:
            self.items[idx] = self.get_from_disk(idx)
        return self.items[idx]

    def fetch_bright_stars(self, po_fits, img, wcs, bg, psf, allow_edge=False):
        is_star = po_fits["objc_type"] == 6
        is_bright = po_fits["psfflux"].sum(axis=1) > 100
        is_thing = po_fits["thing_id"] != -1
        is_target = is_star & is_bright & is_thing
        ras = po_fits["ra"][is_target]
        decs = po_fits["dec"][is_target]
        fluxes = po_fits["psfflux"][is_target].sum(axis=1)

        band = 2
        stamps = []
        pts = []
        prs = []
        bgs = []
        flxs = []

        G = construct_subpixel_grid_base(self.stampsize, self.stampsize)
        for (ra, dec, f, flux) in zip(ras, decs, po_fits["thing_id"][is_target], fluxes):
            # pt = "time" in pixel coordinates
            pt, pr = wcs.wcs_world2pix(ra, dec, 0)
            pti, pri = int(pt + 0.5), int(pr + 0.5)

            row_lower = pri - self.stampsize//2
            row_upper = pri + self.stampsize//2 + 1
            col_lower = pti - self.stampsize//2
            col_upper = pti + self.stampsize//2 + 1
            edge_stamp=False
            if row_lower < 0:
                row_lower = 0
                row_upper = self.stampsize
                edge_stamp=True
            if row_upper > img.shape[0]:
                row_lower = img.shape[0] - self.stampsize
                row_upper = img.shape[0]
                edge_stamp=True
            if col_lower < 0:
                col_lower = 0
                col_upper = self.stampsize
                edge_stamp=True
            if col_upper > img.shape[1]:
                col_lower = img.shape[1] - self.stampsize
                col_upper = img.shape[1]
                edge_stamp=True
            if (not edge_stamp) or allow_edge:
                stamp = img[row_lower:row_upper, col_lower:col_upper]
                if self.center_subpixel:
                    stamp = center_stamp_subpixel(stamp, pt, pr, G)
                stamps.append(stamp)
                pts.append(pt)
                prs.append(pr)
                stamp_bg = bg[row_lower:row_upper, col_lower:col_upper]
                bgs.append(stamp_bg)
                flxs.append(flux)

        return np.asarray(stamps), np.asarray(pts), np.asarray(prs), np.asarray(flxs), np.asarray(bgs)

    def get_from_disk(self, idx, verbose=False):
        if self.rcfgcs[idx] is None:
            return None
        run, camcol, field, gain, po_fits, psf = self.rcfgcs[idx]

        camcol_dir = self.sdss_path.joinpath(str(run), str(camcol))
        field_dir = camcol_dir.joinpath(str(field))

        image_list = []
        background_list = []
        nelec_per_nmgy_list = []
        calibration_list = []
        gain_list = []
        wcs_list = []

        cache_path = field_dir.joinpath("cache.pkl")
        if cache_path.exists() and not self.overwrite_cache:
            ret = pickle.load(cache_path.open("rb+"))
            return ret

        for b, bl in enumerate("ugriz"):
            if not (b in self.bands):
                continue

            frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format(
                bl, run, camcol, field
            )
            frame_path = str(field_dir.joinpath(frame_name))
            if verbose:
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

        stamps, pts, prs, fluxes, stamp_bgs = self.fetch_bright_stars(po_fits, image_list[2], wcs_list[2], background_list[2], psf[2])
        stamp_psfs = np.asarray(psf_at_points(pts, prs, psf[2]))
        if len(stamp_bgs) > 0:
            psf_center = stamp_psfs.shape[-1] // 2
            psf_lower  = psf_center - floor(self.stampsize/2)
            psf_upper  = psf_center + ceil(self.stampsize/2)
            stamp_psfs = stamp_psfs[:, psf_lower:psf_upper, psf_lower:psf_upper]

        ret = {
            "image": np.stack(image_list),
            "field": field,
            "background": np.stack(background_list),
            "nelec_per_nmgy": np.stack(nelec_per_nmgy_list),
            "gain": np.stack(gain_list),
            "calibration": np.stack(calibration_list),
            "wcs": wcs_list,
            "bright_stars": stamps,
            "pts": pts,
            "prs": prs,
            "fluxes": fluxes,
            "bright_star_bgs": stamp_bgs,
            "sdss_psfs" : stamp_psfs
        }
        pickle.dump(ret, cache_path.open("wb+"))

        return ret
