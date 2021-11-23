"""Scripts to produce BLISS estimates on survey images. Currently only SDSS is supported."""
from unicodedata import name
import torch
from typing import Tuple, Dict, List
from collections import namedtuple
from torch import nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.tensor import Tensor
from tqdm import tqdm

from bliss.datasets import sdss
from bliss.models.binary import BinaryEncoder
from bliss.models.encoder import (
    ImageEncoder,
    get_full_params,
    get_is_on_from_n_sources,
    get_star_bool,
)
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.models.galaxy_net import OneCenteredGalaxyDecoder
from bliss.sleep import SleepPhase


class LocationTileMAP:
    def __init__(
        self, ptiles: Tensor, tile_is_on_array: Tensor, tile_map_dict: Dict[str, Tensor]
    ) -> None:
        self.ptiles = ptiles
        self.tile_is_on_array = tile_is_on_array
        self.locs = tile_map_dict["locs"]
        self.log_fluxes = tile_map_dict["log_fluxes"]
        self.fluxes = tile_map_dict["fluxes"]
        self.prob_n_sources = tile_map_dict["prob_n_sources"]
        self.n_sources = tile_map_dict["n_sources"]


class LocationTileVarParams:
    def __init__(self, var_params_dict: Dict[str, Tensor]) -> None:
        self.loc_mean, self.loc_logvar = var_params_dict["loc_mean"], var_params_dict["loc_logvar"]
        self.log_flux_mean, self.log_flux_logvar = (
            var_params_dict["log_flux_mean"],
            var_params_dict["log_flux_logvar"],
        )
        self.n_source_log_probs = var_params_dict["n_source_log_probs"]


class ClassificationTileMAP:
    def __init__(self, galaxy_bool: Tensor, star_bool: Tensor, prob_galaxy: Tensor) -> None:
        self.galaxy_bool = galaxy_bool
        self.star_bool = star_bool
        self.prob_galaxy = prob_galaxy


class TileMAP:
    """Maximum a posteriori estimates on each tile of an image."""

    def __init__(
        self,
        loc_tile_map: LocationTileMAP,
        classification_map: ClassificationTileMAP,
        galaxy_params: Tensor,
    ):
        self.locs = loc_tile_map.locs
        self.log_fluxes = loc_tile_map.log_fluxes
        self.fluxes = loc_tile_map.fluxes
        self.prob_n_sources = loc_tile_map.prob_n_sources
        self.n_sources = loc_tile_map.n_sources

        self.galaxy_bool = classification_map.galaxy_bool
        self.star_bool = classification_map.star_bool
        self.prob_galaxy = classification_map.prob_galaxy

        self.galaxy_params = galaxy_params

    @property
    def n_tiles(self):
        return self.locs.shape[1]

    def asdict(self):
        return self.__dict__


class FullMAP(nn.Module):
    """Maximum a posteriori estimates for each identified object in an image."""

    def __init__(
        self,
        tile_map: TileMAP,
        x: float,
        y: float,
        h: int,
        w: int,
        bp: int,
        galaxy_decoder: OneCenteredGalaxyDecoder = None,
    ) -> None:
        super().__init__()
        full_map_dict = get_full_params(tile_map.asdict(), h - 2 * bp, w - 2 * bp)
        plocs = full_map_dict["plocs"]
        plocs = plocs.reshape(-1, 2)

        plocs_x = plocs[:, 1] + x - 0.5
        plocs_y = plocs[:, 0] + y - 0.5
        self.plocs_xy = torch.stack((plocs_x, plocs_y), dim=-1)
        self.galaxy_bool = full_map_dict["galaxy_bool"].reshape(-1)
        self.prob_galaxy = full_map_dict["prob_galaxy"].reshape(-1)
        self.star_flux = full_map_dict["fluxes"].reshape(-1)

        galaxy_params = full_map_dict["galaxy_params"]
        self.galaxy_params = galaxy_params.reshape(-1, galaxy_params.shape[-1])

        if galaxy_decoder is not None:
            self.fluxes, self.mags = self._get_fluxes_and_mags(galaxy_decoder)
        else:
            self.fluxes = None
            self.mags = None

    @property
    def n_objects(self):
        return self.locs.shape[1]

    def _get_fluxes_and_mags(self, galaxy_decoder: OneCenteredGalaxyDecoder):
        # latent_dim = self.galaxy_params.shape[-1]
        latents = self.galaxy_params
        galaxy_flux = galaxy_decoder(latents).sum((-1, -2, -3)).reshape(-1)
        # collect flux and magnitude into a single tensor
        est_fluxes = self.star_flux * (1 - self.galaxy_bool) + galaxy_flux * self.galaxy_bool
        est_mags = sdss.convert_flux_to_mag(est_fluxes.cpu())
        return est_fluxes, est_mags

    def move_to_cpu(self):
        for name in self.__dict__:
            obj = getattr(self, name)
            if isinstance(obj, Tensor):
                setattr(self, name, obj.cpu())


class TileVarParams:
    """Variational parameters on each tile of an image."""

    def __init__(
        self,
        loc_var_params: LocationTileVarParams,
        classification_map: ClassificationTileMAP,
        galaxy_param_mean: Tensor,
    ) -> None:
        self.loc_mean, self.loc_logvar = loc_var_params.loc_mean, loc_var_params.loc_logvar
        self.log_flux_mean, self.log_flux_logvar = (
            loc_var_params.log_flux_mean,
            loc_var_params.log_flux_logvar,
        )
        self.n_source_log_probs = loc_var_params.n_source_log_probs

        self.galaxy_bool = classification_map.galaxy_bool
        self.star_bool = classification_map.star_bool
        self.prob_galaxy = classification_map.galaxy_bool

        self.galaxy_param_mean = galaxy_param_mean


class Predict(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        binary_encoder: BinaryEncoder,
        galaxy_encoder: GalaxyEncoder,
        galaxy_decoder: OneCenteredGalaxyDecoder,
    ) -> None:
        """Initializes Predict module.

        Args:
            image_encoder: Trained ImageEncoder model.
            galaxy_encoder: Trained GalaxyEncoder model.
            binary_encoder: Trained BinaryEncoder model.
            galaxy_decoder: Trained OneCenteredGalaxyDecoder model.

        """
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.image_encoder = image_encoder
        self.binary_encoder = binary_encoder
        self.galaxy_encoder = galaxy_encoder
        self.galaxy_decoder = galaxy_decoder

    def forward(self, x):
        pass

    @property
    def device(self):
        return self._dummy_param.device

    @property
    def latent_dim(self):
        return self.galaxy_encoder.latent_dim

    def predict_on_image(
        self,
        image: torch.Tensor,
    ) -> Tuple[TileMAP, TileVarParams]:
        """This function takes in a single image and outputs the prediction from trained models.

        Prediction requires counts/locations provided by a trained `image_encoder` that is
        set upon initializing the module.
        Note that prediction is done on a central square of the image
        corresponding to a border of size `image_encoder.border_padding`.

        Args:
            image: Tensor of shape (1, n_bands, h, w) where slen-2*border_padding <= 300.

        Returns:
            tile_map: Object with MAP estimates for parameters of sources in each tile.
            full_map: Object with MAP estimates for parameters of sources on full image.
            tile_var_params: Object containing tensors of variational parameters corresponding
                to each tile in `image`.
        """
        self._validate_image(image)

        # prepare dimensions

        loc_tile_map, loc_var_params = self.locate_objects(image)
        classification_map = self.classify_objects(loc_tile_map)
        galaxy_param_mean = self.get_galaxy_params(loc_tile_map, classification_map)

        tile_map = TileMAP(loc_tile_map, classification_map, galaxy_param_mean)
        tile_var_params = TileVarParams(loc_var_params, classification_map, galaxy_param_mean)
        # full parameters on chunk

        return tile_map, tile_var_params

    def locate_objects(self, image) -> Tuple[LocationTileMAP, LocationTileVarParams]:
        # get padded tiles.
        ptiles = self.image_encoder.get_images_in_tiles(image)

        # get MAP estimates and variational parameters from image_encoder
        tile_n_sources = self.image_encoder.tile_map_n_sources(ptiles)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, 1).reshape(1, -1, 1, 1)
        tile_map = self.image_encoder.tile_map_estimate(image)
        var_params = self.image_encoder(ptiles, tile_n_sources)

        loc_tile_map = LocationTileMAP(ptiles, tile_is_on_array, tile_map)
        loc_var_params = LocationTileVarParams(var_params)
        return loc_tile_map, loc_var_params

    # def classify_objects(self, ptiles, locs, tile_is_on_array, n_sources):
    def classify_objects(self, loc_tile_map: LocationTileMAP):
        assert not self.binary_encoder.training
        prob_galaxy = (
            self.binary_encoder(loc_tile_map.ptiles, loc_tile_map.locs).reshape(1, -1, 1, 1)
            * loc_tile_map.tile_is_on_array
        )
        galaxy_bool = (prob_galaxy > 0.5).float() * loc_tile_map.tile_is_on_array
        star_bool = get_star_bool(loc_tile_map.n_sources, galaxy_bool)
        return ClassificationTileMAP(galaxy_bool, star_bool, prob_galaxy)

    def get_galaxy_params(
        self, loc_tile_map: LocationTileMAP, classification_map: ClassificationTileMAP
    ):
        galaxy_param_mean = self.galaxy_encoder(loc_tile_map.ptiles, loc_tile_map.locs)
        latent_dim = galaxy_param_mean.shape[-1]
        galaxy_param_mean = galaxy_param_mean.reshape(1, -1, 1, latent_dim)
        galaxy_param_mean *= loc_tile_map.tile_is_on_array * classification_map.galaxy_bool
        return galaxy_param_mean

    def predict_on_scene(
        self,
        clen: int,
        scene: torch.Tensor,
        device: torch.device = "cpu",
        testing=False,
    ):
        """Perform predictions chunk-by-chunk when image is larger than 300x300 pixels.

        The scene will be divided up into chunks of side length `clen`. Prediction will be
        done in every part of the scene except for a border of length
        `image_encoder.border_padding`.
        To be more specific, any sources with centroids (x0, y0) satisfying any of the following
        conditions: ``0 < x0 < bp``, ``w - bp < x0 < w``, ``0 < y0 < bp``, ``h - bp < y0 < h``
        will NOT be detected by our models.

        Args:
            clen: Dimensions of (unpadded) chunks we want to extract from scene.
            scene: Tensor of shape (1, n_bands, h, w) containing image of scene we will make
                predictions.
            device: Device where each model is currently and where padded chunks will be moved.
            testing: Whether we are unit testing and we only want to run 1 chunk.

        Returns:
            results: List containing the results of prediction on each chunk, i.e. tuples of
                `tile_map, full_map, var_params` as returned by `predict_on_image`.
        """
        assert len(scene.shape) == 4
        assert scene.shape[0] == 1
        assert scene.shape[1] == self.image_encoder.n_bands == 1, "Only 1 band supported"
        h, w = scene.shape[-2], scene.shape[-1]
        bp = self.image_encoder.border_padding
        ihic = h // clen + 1 if not testing else 1  # height in chunks
        iwic = w // clen + 1 if not testing else 1  # width in chunks
        self.to(device)

        # tiles
        tile_slen = self.image_encoder.tile_slen

        # where to collect results.
        var_params = []
        locs = torch.tensor([])
        galaxy_bool = torch.tensor([])
        prob_galaxy = torch.tensor([])
        fluxes = torch.tensor([])
        mags = torch.tensor([])

        with torch.no_grad():
            with tqdm(total=ihic * iwic) as pbar:
                for i in range(iwic):
                    for j in range(ihic):
                        x1, y1 = i * clen + bp, j * clen + bp

                        # the following two statements ensure divisibility by tile_slen
                        # of the resulting chunk near the edges.
                        if clen + bp > w - x1:
                            x1 = x1 + (w - x1 - bp) % tile_slen

                        if clen + bp > h - y1:
                            y1 = y1 + (h - y1 - bp) % tile_slen

                        pchunk = scene[:, :, y1 - bp : y1 + clen + bp, x1 - bp : x1 + clen + bp]
                        pchunk = pchunk.to(self.device)

                        tile_map, vparams = self.predict_on_image(pchunk)
                        h_pchunk, w_pchunk = pchunk.shape[-2], pchunk.shape[-1]
                        full_map = FullMAP(
                            tile_map, x1, y1, h_pchunk, w_pchunk, bp, self.galaxy_decoder
                        )
                        full_map.move_to_cpu()

                        # delete parameters we stopped using so we have enough GPU space.
                        if "cuda" in self.device.type:
                            del pchunk
                            torch.cuda.empty_cache()

                        # concatenate to obtain tensors on full image.
                        locs = torch.cat((locs, full_map.plocs_xy))
                        galaxy_bool = torch.cat((galaxy_bool, full_map.galaxy_bool))
                        prob_galaxy = torch.cat((prob_galaxy, full_map.prob_galaxy))
                        fluxes = torch.cat((fluxes, full_map.fluxes))
                        mags = torch.cat((mags, full_map.mags))

                        # save variational parameters
                        var_params.append(vparams)

                        # update progress bar
                        pbar.update(1)

        full_map = {
            "plocs": locs,
            "galaxy_bool": galaxy_bool,
            "prob_galaxy": prob_galaxy,
            "flux": fluxes,
            "mag": mags,
            "n_sources": torch.tensor([len(locs)]),
        }

        return var_params, full_map

    def _validate_image(self, image):
        # prepare and check consistency
        assert not self.image_encoder.training
        assert len(image.shape) == 4
        assert image.shape[0] == 1
        assert image.shape[1] == self.image_encoder.n_bands == 1
        assert self.image_encoder.max_detections == 1
        # binary prediction
        assert not self.binary_encoder.training
        assert image.shape[1] == self.binary_encoder.n_bands
        # galaxy measurement predictions
        assert not self.galaxy_encoder.training
        assert image.shape[1] == self.galaxy_encoder.n_bands
        assert self.image_encoder.border_padding == self.galaxy_encoder.border_padding
        assert self.image_encoder.tile_slen == self.galaxy_encoder.tile_slen


def predict(cfg: DictConfig):
    bands = list(cfg.predict.bands)
    print("-" * 20 + " Predicting Configuration " + "-" * 20)
    print(OmegaConf.to_yaml(cfg.predict))
    assert isinstance(bands, list) and len(bands) == 1, "Only 1 band supported"

    # setup params from config
    clen = cfg.predict.clen
    device = torch.device(cfg.predict.device)
    testing = cfg.predict.testing

    # load images from SDSS for prediction.
    sdss_obj = sdss.SloanDigitalSkySurvey(**cfg.predict.sdss_kwargs)
    image = sdss_obj[0]["image"][0]
    image = rearrange(torch.from_numpy(image), "h w -> 1 1 h w")

    # load models.
    sleep_net = SleepPhase.load_from_checkpoint(cfg.predict.sleep_checkpoint)
    galaxy_encoder = GalaxyEncoder.load_from_checkpoint(cfg.predict.galaxy_checkpoint)
    binary_encoder = BinaryEncoder.load_from_checkpoint(cfg.predict.binary_checkpoint)

    # move everything to specified GPU
    image_encoder = sleep_net.image_encoder.eval().to(device)
    binary_encoder = binary_encoder.to(device).eval()
    galaxy_encoder = galaxy_encoder.to(device).eval()
    galaxy_decoder = sleep_net.image_decoder.galaxy_tile_decoder.galaxy_decoder.eval().to(device)

    predict_module = Predict(image_encoder, binary_encoder, galaxy_encoder, galaxy_decoder)
    var_params, _ = predict_module.predict_on_scene(clen, image, device, testing)

    if cfg.predict.output_file is not None:
        torch.save(var_params, cfg.predict.output_file)
        print(f"Prediction saved to {cfg.predict.output_file}")
