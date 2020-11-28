import numpy as np
import torch
from astropy.table import Table
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler
from omegaconf import DictConfig


n_bands_dict = {1: ("i",), 6: ("y", "z", "i", "r", "g", "u")}


def filter_bounds(cat, param, min_val=-np.inf, max_val=np.inf):
    return cat[(cat[param] >= min_val) & (cat[param] <= max_val)]


class SurveyObs(object):
    def __init__(self, renderer):
        """Returns a list of :class:`Survey` objects, each of them has an image attribute which is
        where images are written to by iso_render_engine.render_galaxy.
        """
        import descwl

        self.pixel_scale = renderer.pixel_scale
        self.dtype = renderer.dtype

        obs = []
        for band in renderer.bands:
            # dictionary of default values.
            survey_dict = descwl.survey.Survey.get_defaults(
                survey_name=renderer.survey_name, filter_band=band
            )

            assert (
                renderer.pixel_scale == survey_dict["pixel_scale"]
            ), "Pixel scale does not match particular band?"
            survey_dict["image_width"] = renderer.slen  # pixels
            survey_dict["image_height"] = renderer.slen

            descwl_survey = descwl.survey.Survey(
                no_analysis=True,
                survey_name=renderer.survey_name,
                filter_band=band,
                **survey_dict,
            )
            obs.append(descwl_survey)

        self.obs = obs

    def __enter__(self):
        return self.obs

    def __exit__(self, exc_type, exc_val, exc_tb):
        import galsim

        for single_obs in self.obs:
            single_obs.image = galsim.Image(
                bounds=single_obs.image_bounds,
                scale=self.pixel_scale,
                dtype=self.dtype,
            )


class CatsimRenderer(object):
    def __init__(
        self,
        survey_name="LSST",
        n_bands=1,
        slen=41,
        snr=200,
        min_snr=0.05,
        truncate_radius=30,
        add_noise=True,
        preserve_flux=False,
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
        assert survey_name == "LSST", "Only using default survey name for now is LSST."
        assert n_bands in [1, 6], "Only 1 or 6 bands are supported."
        assert add_noise, "Are you sure?"
        assert slen >= 41, "Galaxies won't fit."
        assert slen % 2 == 1, "Odd number of pixels is preferred."
        assert preserve_flux is False, "Otherwise variance of the noise will change."

        self.survey_name = survey_name
        self.n_bands = n_bands
        self.bands = n_bands_dict[n_bands]

        self.slen = slen
        self.pixel_scale = self.get_pixel_scale(self.survey_name)
        stamp_size = self.pixel_scale * self.slen  # arcseconds
        self.slen = int(stamp_size / self.pixel_scale)  # pixels.

        self.snr = snr
        self.min_snr = min_snr
        self.truncate_radius = truncate_radius

        self.add_noise = add_noise
        self.deviate_center = deviate_center
        self.preserve_flux = preserve_flux  # when changing SNR.
        self.verbose = verbose
        self.dtype = dtype

        self.background = self.get_background()

    def get_background(self):
        background = np.zeros((self.n_bands, self.slen, self.slen), dtype=self.dtype)

        with SurveyObs(self) as obs:
            for i, single_obs in enumerate(obs):
                background[i, :, :] = single_obs.mean_sky_level
        return background

    def center_deviation(self, entry):
        # random deviation from exactly in center of center pixel, in arcseconds.
        deviation_ra = (np.random.rand() - 0.5) if self.deviate_center else 0.0
        deviation_dec = (np.random.rand() - 0.5) if self.deviate_center else 0.0
        entry["ra"] = deviation_ra * self.pixel_scale  # arcseconds
        entry["dec"] = deviation_dec * self.pixel_scale
        return entry

    def render(self, entry):
        """
        Return a multi-band image corresponding to the entry from the catalog given.

        * The final image includes its background based on survey's sky level.
        * If deviate_center==True, then galaxy not aligned between bands.
        """
        image = np.zeros((self.n_bands, self.slen, self.slen), dtype=self.dtype)

        with SurveyObs(self) as obs:
            for i, band in enumerate(self.bands):
                entry = self.center_deviation(entry)
                image_no_background = self.single_band(entry, obs[i], band)
                image[i, :, :] = image_no_background + self.background[i]

        return image, self.background

    def single_band(self, entry, single_obs, band):
        """Builds galaxy from a single entry in the catalog. With no background sky level added."""
        import descwl
        import galsim

        galaxy_builder = descwl.model.GalaxyBuilder(
            single_obs, no_disk=False, no_bulge=False, no_agn=False, verbose_model=False
        )

        galaxy = galaxy_builder.from_catalog(entry, entry["ra"], entry["dec"], band)

        iso_render_engine = descwl.render.Engine(
            survey=single_obs,
            min_snr=self.min_snr,
            truncate_radius=self.truncate_radius,
            no_margin=False,
            verbose_render=False,
        )

        # Up to this point, single_obs has not been changed by the previous 3 statements.

        try:
            # this line draws the given galaxy image onto single_obs.image,
            # this is the only change in single_obs.
            iso_render_engine.render_galaxy(
                galaxy,
                variations_x=None,
                variations_s=None,
                variations_g=None,
                no_fisher=True,
                calculate_bias=False,
                no_analysis=True,
            )  # saves image in single_obs

        except descwl.render.SourceNotVisible:
            if self.verbose:
                print("Source not visible")
            raise descwl.render.SourceNotVisible  # pass it on with a warning.

        image_temp = galsim.Image(self.slen, self.slen)
        image_temp += single_obs.image

        if self.add_noise:
            # NOTE: PoissonNoise assumes background already subtracted off.
            generator = galsim.random.BaseDeviate(seed=np.random.randint(99999999))
            noise = galsim.PoissonNoise(
                rng=generator, sky_level=single_obs.mean_sky_level
            )

            # Both of the adding noise methods add noise on the image consisting of the
            # (galaxy flux + background), but then remove the background at the end so we need
            # to add it later.
            if self.snr:
                image_temp.addNoiseSNR(
                    noise, snr=self.snr, preserve_flux=self.preserve_flux
                )
            else:
                image_temp.addNoise(noise)

        return image_temp.array

    @staticmethod
    def get_pixel_scale(survey_name):
        import descwl

        return descwl.survey.Survey.get_defaults(survey_name, "*")["pixel_scale"]


# TODO: Make this Dataset/Renderer faster, still can't do on the fly.
class CatsimGalaxies(pl.LightningDataModule, Dataset):
    def __init__(self, cfg: DictConfig):
        """This class reads a random entry from the OneDegSq.fits file (sample from the Catsim catalog)
        and returns a galaxy drawn from the catalog with realistic seeing conditions using
        functions from WeakLensingDeblending.
        """
        super(CatsimGalaxies, self).__init__()
        self.renderer = CatsimRenderer(**cfg.dataset.renderer)
        self.background = self.renderer.background

        # prepare catalog table.
        # shuffle in case that order matters.
        cat = Table.read(cfg.dataset.catalog_file)
        cat = cat[np.random.permutation(len(cat))]
        self.filter_dict = self.get_default_filters()
        self.cat = self.get_filtered_cat(cat)

        # data processing
        self.num_workers = cfg.dataset.num_workers
        self.batch_size = cfg.dataset.batch_size
        self.n_batches = cfg.dataset.n_batches
        n_samples = self.batch_size * self.n_batches
        self.sampler = RandomSampler(self, replacement=True, num_samples=n_samples)

    @staticmethod
    def get_default_filters():
        # cut on magnitude same as BTK does (gold sample)
        filters = dict(i_ab=(-np.inf, 25.3))
        return filters

    def get_filtered_cat(self, cat):
        _cat = cat.copy()
        for param, bounds in self.filter_dict.items():
            min_val, max_val = bounds
            _cat = filter_bounds(_cat, param, min_val, max_val)
        return _cat

    def __len__(self):
        return len(self.cat)

    def __getitem__(self, idx):
        import descwl

        while True:  # loop until visible galaxy is selected.
            try:
                entry = self.cat[idx]
                image, background = self.renderer.render(entry)
                break

            # select some other random galaxy to return if we fail.
            except descwl.render.SourceNotVisible:
                idx = np.random.choice(np.arange(len(self)))

        return {"images": image, "background": background}

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
        super(SavedCatsim, self).__init__()
        params = cfg.dataset.params
        self.data = torch.load(params.filepath)
        self.images = self.data["images"]
        assert isinstance(self.images, torch.Tensor)
        assert len(self.images.shape) == 4

        background = self.data.pop("background")
        assert len(background.shape) == 4
        self.background = background[0]

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
