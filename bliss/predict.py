"""Scripts to produce BLISS estimates on survey images. Currently only SDSS is supported."""
import torch
from torch import nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
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
    ):
        """This function takes in a single image and outputs the prediction from trained models.

        Prediction requires counts/locations provided by a trained `image_encoder` so this is
        a required argument. Note that prediction is done on a central square of the image
        corresponding to a border of size `image_encoder.border_padding`.

        Args:
            image: Tensor of shape (1, n_bands, h, w) where slen-2*border_padding <= 300.

        Returns:
            tile_map: Dictionary containing MAP estimates for parameters of sources in each tile.
            full_map: Dictionary containing MAP estimates for parameters of sources on full image.
            var_params: Dictionary containing tensors of variational parameters corresponding
                to each tile in `image`. The variational parameters include `prob_galaxy`,
                `prob_n_sources`, `loc_mean`, `loc_logvar`, etc.
        """
        # prepare and check consistency
        assert not self.image_encoder.training
        assert len(image.shape) == 4
        assert image.shape[0] == 1
        assert image.shape[1] == self.image_encoder.n_bands == 1
        assert self.image_encoder.max_detections == 1

        # prepare dimensions
        h, w = image.shape[-2], image.shape[-1]
        bp = self.image_encoder.border_padding

        ptiles, tile_is_on_array, var_params, tile_map = self.locate_objects(image)

        # binary prediction
        assert not self.binary_encoder.training
        assert image.shape[1] == self.binary_encoder.n_bands

        prob_galaxy = (
            self.binary_encoder(ptiles, tile_map["locs"]).reshape(1, -1, 1, 1) * tile_is_on_array
        )
        galaxy_bool = (prob_galaxy > 0.5).float() * tile_is_on_array
        star_bool = get_star_bool(tile_map["n_sources"], galaxy_bool)
        var_params["galaxy_bool"] = galaxy_bool
        var_params["star_bool"] = star_bool
        var_params["prob_galaxy"] = prob_galaxy
        tile_map["galaxy_bool"] = galaxy_bool
        tile_map["star_bool"] = star_bool
        tile_map["prob_galaxy"] = prob_galaxy

        # get galaxy measurement predictions
        assert not self.galaxy_encoder.training
        assert image.shape[1] == self.galaxy_encoder.n_bands
        assert self.image_encoder.border_padding == self.galaxy_encoder.border_padding
        assert self.image_encoder.tile_slen == self.galaxy_encoder.tile_slen

        galaxy_param_mean = self.galaxy_encoder(ptiles, tile_map["locs"])
        latent_dim = galaxy_param_mean.shape[-1]
        galaxy_param_mean = galaxy_param_mean.reshape(1, -1, 1, latent_dim)
        galaxy_param_mean *= tile_is_on_array * galaxy_bool
        var_params["galaxy_param_mean"] = galaxy_param_mean
        tile_map["galaxy_params"] = galaxy_param_mean

        # full parameters on chunk
        full_map = get_full_params(tile_map, h - 2 * bp, w - 2 * bp)

        return tile_map, full_map, var_params

    def locate_objects(self, image):
        # get padded tiles.
        ptiles = self.image_encoder.get_images_in_tiles(image)

        # get MAP estimates and variational parameters from image_encoder
        tile_n_sources = self.image_encoder.tile_map_n_sources(ptiles)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, 1).reshape(1, -1, 1, 1)
        tile_map = self.image_encoder.tile_map_estimate(image)
        var_params = self.image_encoder(ptiles, tile_n_sources)

        return ptiles, tile_is_on_array, var_params, tile_map

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

                        _, est_params, vparams = self.predict_on_image(pchunk)

                        (
                            est_locs,
                            est_gbool,
                            est_pgalaxy,
                            est_star_flux,
                            est_galaxy_params,
                        ) = self._process_est_params(est_params, x1, y1)

                        # delete parameters we stopped using so we have enough GPU space.
                        if "cuda" in self.device.type:
                            del est_params
                            del pchunk
                            torch.cuda.empty_cache()

                        est_fluxes, est_mags = self._get_fluxes_and_mags(
                            est_gbool, est_star_flux, est_galaxy_params
                        )

                        # concatenate to obtain tensors on full image.
                        locs = torch.cat((locs, est_locs))
                        galaxy_bool = torch.cat((galaxy_bool, est_gbool))
                        prob_galaxy = torch.cat((prob_galaxy, est_pgalaxy))
                        fluxes = torch.cat((fluxes, est_fluxes))
                        mags = torch.cat((mags, est_mags))

                        # save variational parameters
                        var_params.append({key: value.cpu() for key, value in vparams.items()})

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

        return vparams, full_map

    def _process_est_params(self, est_params, x1, y1):
        est_locs = est_params["plocs"].cpu().reshape(-1, 2)
        est_gbool = est_params["galaxy_bool"].cpu().reshape(-1)
        est_pgalaxy = est_params["prob_galaxy"].cpu().reshape(-1)
        est_star_flux = est_params["fluxes"].cpu().reshape(-1)
        est_galaxy_params = est_params["galaxy_params"].cpu().reshape(-1, self.latent_dim)

        # locations in pixels consistent with full scene.
        # 0.5 comes from pt/pr definition (plotting both -> off by half a pixel).
        x = est_locs[:, 1].reshape(-1, 1) + x1 - 0.5
        y = est_locs[:, 0].reshape(-1, 1) + y1 - 0.5
        est_locs = torch.hstack((x, y)).reshape(-1, 2)

        return est_locs, est_gbool, est_pgalaxy, est_star_flux, est_galaxy_params

    def _get_fluxes_and_mags(self, est_gbool, est_star_flux, est_galaxy_params):
        latents = est_galaxy_params.to(self.device).reshape(-1, self.latent_dim)
        est_galaxy_flux = self.galaxy_decoder(latents).sum((-1, -2, -3)).cpu().reshape(-1)
        # collect flux and magnitude into a single tensor
        est_fluxes = est_star_flux * (1 - est_gbool) + est_galaxy_flux * est_gbool
        est_mags = sdss.convert_flux_to_mag(est_fluxes)
        return est_fluxes, est_mags


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
