import numpy as np
from pathlib import Path
from astropy.table import Table
from numpy.lib import npyio
from omegaconf import DictConfig
from astropy.io import fits
import galsim

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler
import pytorch_lightning as pl


n_bands_dict = {1: ("r",), 6: ("y", "z", "i", "r", "g", "u")}

SDSS_PIX = 0.396


class SourceNotVisible(Exception):
    """Custom exception to indicate that a source has no visible model components."""


class CatsimRenderer:
    def __init__(
        self,
        survey_name="SDSS",
        n_bands=1,
        slen=41,
        background=(646,),  # r-band
        snr=200,
        psf_file="data/sdss-002583-2-0136-psf-r.fits",
        pixel_scale=SDSS_PIX,
        add_noise=True,
        verbose=False,
        deviate_center=False,
        dtype=np.float32,
    ):
        """
        Can draw a single entry in CATSIM in the given bands.

        NOTE: Background is constant given the band, survey_name, image size, and default
        survey_dict, so it can be obtained in advance only once.
        """
        # ToDo: Create a test/assertion to check that mean == variance approx.
        assert survey_name == "SDSS", "Only using default survey name for now is SDSS."
        assert n_bands == 1, "Only 1 band is supported"
        # assert slen >= 41, "Galaxies won't fit."
        assert slen % 2 == 1, "Odd number of pixels is preferred."
        assert Path(psf_file).is_file(), "psf file not found."
        assert Path(psf_file).as_posix().endswith(".fits")

        self.survey_name = survey_name
        self.n_bands = n_bands
        self.bands = n_bands_dict[n_bands]

        self.slen = slen
        self.pixel_scale = pixel_scale
        stamp_size = self.pixel_scale * self.slen  # arcseconds
        self.slen = int(stamp_size / self.pixel_scale)  # pixels.

        self.snr = snr

        self.add_noise = add_noise
        self.deviate_center = deviate_center
        self.verbose = verbose
        self.dtype = dtype

        # get background
        image_shape = (self.n_bands, self.slen, self.slen)
        self.background = np.zeros(image_shape, dtype=self.dtype)
        for b in range(self.n_bands):
            self.background[b, :, :] = background[b]

        # Get unintegrated galsim psf for the convolution
        psf = fits.getdata(psf_file)
        assert len(psf.shape) == 2, "Same PSF for all bands for now."
        psf = psf - psf[1, int(psf.shape[1] / 2)] * 2
        psf[psf < 0] = 0
        psf /= np.sum(psf)
        self.psf = galsim.InterpolatedImage(
            galsim.Image(psf), scale=self.pixel_scale
        ).withFlux(1.0)

    def center_deviation(self, entry):
        # random deviation from exactly in center of center pixel, in arcseconds.
        deviation_ra = (np.random.rand() - 0.5) if self.deviate_center else 0.0
        deviation_dec = (np.random.rand() - 0.5) if self.deviate_center else 0.0
        entry["ra"] = deviation_ra * self.pixel_scale
        entry["dec"] = deviation_dec * self.pixel_scale
        return entry

    def render(self, entry):
        """
        Return a multi-band image corresponding to the entry from the catalog given.

        * The final image includes background.
        * If deviate_center==True, then galaxy not aligned between bands.
        """
        image = np.zeros((self.n_bands, self.slen, self.slen), dtype=self.dtype)
        entry = self.center_deviation(entry)  # all bands are centered same way.

        for b, band in enumerate(self.bands):
            image[b] = self.single_band_galaxy(entry, band).array

        # add background and (optionally) Gaussian noise
        image += self.background
        if self.add_noise:
            _image = np.sqrt(image)
            _image = _image * np.random.randn(*image.shape) * 0.1
            image += _image
        return image, self.background

    def get_flux(self, ab_magnitude):
        """Convert source magnitude to flux.
        Args:
            ab_magnitude(float): AB magnitude of source.
        Returns:
            float: Flux in detected electrons.
        """

        # TODO: other bands that are not 'r'
        return 3000 * 10 ** (-0.4 * (ab_magnitude - 24))

    def single_band_galaxy(
        self, entry, band, no_disk=False, no_bulge=False, no_agn=False
    ):
        """Builds galaxy from a single entry in the catalog. With background and noise"""
        components = []

        mag_name = band + "_ab"
        total_flux = self.get_flux(entry[mag_name])

        # Calculate the flux of each component in detected electrons.
        total_fluxnorm = (
            entry["fluxnorm_disk"] + entry["fluxnorm_bulge"] + entry["fluxnorm_agn"]
        )
        disk_flux = (
            0.0 if no_disk else entry["fluxnorm_disk"] / total_fluxnorm * total_flux
        )
        bulge_flux = (
            0.0 if no_bulge else entry["fluxnorm_bulge"] / total_fluxnorm * total_flux
        )
        agn_flux = (
            0.0 if no_agn else entry["fluxnorm_agn"] / total_fluxnorm * total_flux
        )

        if disk_flux + bulge_flux + agn_flux == 0:
            raise SourceNotVisible

        # Calculate the position of angle of the Sersic components, which are assumed to be the same.
        if disk_flux > 0:
            beta_radians = np.radians(entry["pa_disk"])
            if bulge_flux > 0:
                assert (
                    entry["pa_disk"] == entry["pa_bulge"]
                ), "Sersic components have different beta."
        elif bulge_flux > 0:
            beta_radians = np.radians(entry["pa_bulge"])
        else:
            # This might happen if we only have an AGN component.
            beta_radians = None
        # Calculate shapes hlr = sqrt(a*b) and q = b/a of Sersic components.
        if disk_flux > 0:
            a_d, b_d = entry["a_d"], entry["b_d"]
            disk_hlr_arcsecs = np.sqrt(a_d * b_d)
            disk_q = b_d / a_d
        else:
            disk_hlr_arcsecs, disk_q = None, None
        if bulge_flux > 0:
            a_b, b_b = entry["a_b"], entry["b_b"]
            bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
            bulge_q = b_b / a_b
        else:
            bulge_hlr_arcsecs, bulge_q = None, None

        if disk_flux > 0:
            disk = galsim.Exponential(
                flux=disk_flux, half_light_radius=disk_hlr_arcsecs
            ).shear(q=disk_q, beta=beta_radians * galsim.radians)
            components.append(disk)

        a_b, b_b = entry["a_b"], entry["b_b"]
        bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
        bulge_q = b_b / a_b

        if disk_flux > 0:
            disk = galsim.Exponential(
                flux=disk_flux, half_light_radius=disk_hlr_arcsecs
            ).shear(q=disk_q, beta=beta_radians * galsim.radians)
            components.append(disk)

        if bulge_flux > 0:
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs
            ).shear(q=bulge_q, beta=beta_radians * galsim.radians)
            components.append(bulge)

        if agn_flux > 0:
            agn = galsim.Gaussian(flux=agn_flux, sigma=1e-8)
            components.append(agn)

        image_temp = galsim.Image(self.slen, self.slen, scale=self.pixel_scale)
        profile = galsim.Add(components)
        profile = galsim.convolve.Convolution(
            profile, self.psf, gsparams=galsim.GSParams(maximum_fft_size=1 << 16)
        )
        image_temp = profile.drawImage(
            image=image_temp,
            use_true_center=True,
            method="auto",
            dtype=self.dtype,
        )
        return image_temp


# TODO: Make this Dataset/Renderer faster, still can't do on the fly.
class CatsimGalaxies(pl.LightningDataModule, Dataset):
    def __init__(self, cfg: DictConfig):
<<<<<<< HEAD
        """This class reads a random entry from the OneDegSq.fits file (sample from the Catsim
        catalog) and returns a galaxy drawn from the catalog with realistic seeing conditions using
        functions from WeakLensingDeblending.
        """
=======
        """This class reads a random entry from the OneDegSq.fits file (sample from the Catsim catalog) and returns a centered galaxy drawn from the catalog."""
>>>>>>> updating catsim so it does not depend on descwl.
        super().__init__()
        self.renderer = CatsimRenderer(**cfg.dataset.renderer)
        self.background = self.renderer.background

        # prepare catalog table.
        # shuffle in case that order matters.
        cat = Table.read(cfg.dataset.catalog_file)
        cat = cat[np.random.permutation(len(cat))]
        self.cat = self.get_filtered_cat(cat)

        # data processing
        self.num_workers = cfg.dataset.num_workers
        self.batch_size = cfg.dataset.batch_size
        self.n_batches = cfg.dataset.n_batches
        n_samples = self.batch_size * self.n_batches
        self.sampler = RandomSampler(self, replacement=True, num_samples=n_samples)

    def get_filtered_cat(self, cat):
        filters = {"i_ab": (-np.inf, 25.3)}
        for param, bounds in filters.items():
            min_val, max_val = bounds
            cat = cat[(cat[param] >= min_val) & (cat[param] <= max_val)]
        return cat

    def __len__(self):
        return len(self.cat)

    def __getitem__(self, idx):

        while True:
            try:
                entry = self.cat[idx]
                image, background = self.renderer.render(entry)
                break
            except SourceNotVisible:
                idx = np.random.choice(range(len(self)))

        return {
            "images": image,
            "background": background,
            "id": int(entry["galtileid"]),
        }

    def train_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
        )


class SavedCatsim(pl.LightningDataModule, Dataset):
    def __init__(self, cfg):
        super().__init__()
        params = cfg.dataset.params
        self.data = torch.load(params.filepath)
        self.images = self.data["images"]
        assert isinstance(self.images, torch.Tensor)
        assert len(self.images.shape) == 4

        self.background = self.data.pop("background")
        assert len(self.background.shape) == 3
        assert self.background.shape[0] == self.images.shape[1]

        self.batch_size = params.batch_size
        self.num_workers = params.num_workers
        self.tt_split = params.tt_split

    def __getitem__(self, idx):
        return {"images": self.images[idx], "background": self.background}

    def __len__(self):
        return len(self.images)

    def train_dataloader(self):
        split = int(len(self) * self.tt_split)
        train_indices = np.arange(split, len(self), dtype=int)
        return DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=SubsetRandomSampler(train_indices),
        )

    def val_dataloader(self):
        split = int(len(self) * self.tt_split)
        test_indices = np.arange(split, dtype=int)
        return DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=SubsetRandomSampler(test_indices),
        )
