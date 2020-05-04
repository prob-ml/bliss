import torch
import torch.nn as nn

from ..utils import const
from ..utils import image_utils


class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)


class Normalize2d(nn.Module):
    def forward(self, tensor):
        assert len(tensor.shape) == 4
        mean = (
            tensor.view(tensor.shape[0], tensor.shape[1], -1)
            .mean(2, keepdim=True)
            .unsqueeze(-1)
        )
        var = (
            tensor.view(tensor.shape[0], tensor.shape[1], -1)
            .var(2, keepdim=True)
            .unsqueeze(-1)
        )

        return (tensor - mean) / torch.sqrt(var + 1e-5)


class SourceEncoder(nn.Module):
    def __init__(
        self,
        slen,
        patch_slen,
        step,
        edge_padding,
        n_bands,
        max_detections,
        n_source_params,
    ):
        """
        This class implements the source encoder, which is supposed to take in a synthetic image of size slen * slen
        and returns a NN latent variable representation of this image.

        * NOTE: Assumes that `source_params` are always `log_fluxes` throughout the code.

        * NOTE: Should have (n_bands == n_source_params) in the case of stars.

        * EXAMPLE on padding: If the patch_slen=8, edge_padding=3, then the size of a tile will be 8-2*3=2

        :param slen: dimension of full image, we assume its square for now
        :param patch_slen: dimension (in pixels) of the individual
                           image patches (usually 8 for stars, and _ for galaxies).
        :param step: number of pixels to shift every subimage/patch.
        :param edge_padding: length of padding (in pixels).
        :param n_bands : number of bands
        :param max_detections:
        :param n_source_params:
        * The dimension of 'source parameters' which are log fluxes in the case of stars and
        latent variable dimension in the case of galaxy. Assumed to be normally distributed.
        * For fluxes this should equal number of bands, for galaxies it will be the number of latent dimensions in the
        network.
        """
        super(SourceEncoder, self).__init__()

        # image parameters
        self.slen = slen
        self.patch_slen = patch_slen
        self.step = step
        self.n_bands = n_bands

        self.edge_padding = edge_padding

        self.tile_coords = image_utils.get_tile_coords(
            self.slen, self.slen, self.patch_slen, self.step
        )
        self.n_patches = self.tile_coords.shape[0]

        # max number of detections
        self.max_detections = max_detections

        # convolutional NN parameters
        enc_conv_c = 20
        enc_kern = 3
        enc_hidden = 256

        momentum = 0.5

        # convolutional NN
        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.n_bands, enc_conv_c, enc_kern, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern, stride=1, padding=1),
            nn.BatchNorm2d(enc_conv_c, momentum=momentum, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern, stride=1, padding=1),
            nn.BatchNorm2d(enc_conv_c, momentum=momentum, track_running_stats=True),
            nn.ReLU(),
            Flatten(),
        )

        # output dimension of convolutions
        conv_out_dim = self.enc_conv(
            torch.zeros(1, n_bands, patch_slen, patch_slen)
        ).size(1)

        # fully connected layers
        self.enc_fc = nn.Sequential(
            nn.Linear(conv_out_dim, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum=momentum, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum=momentum, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum=momentum, track_running_stats=True),
            nn.ReLU(),
        )

        # there are self.max_detections * (self.max_detections + 1)
        #    total possible detections, and each detection has
        #    4 + 2*n parameters (2 means and 2 variances for each loc and for n source_param's
        #    (flux per band or otherwise)
        self.n_source_params = n_source_params
        self.n_params_per_source = 4 + 2 * self.n_source_params

        self.dim_out_all = int(
            0.5
            * self.max_detections
            * (self.max_detections + 1)
            * self.n_params_per_source
            + 1
            + self.max_detections
        )
        self._get_hidden_indices()

        self.enc_final = nn.Linear(enc_hidden, self.dim_out_all)
        self.log_softmax = nn.LogSoftmax(dim=1)

    ############################
    # The layers of our neural network
    ############################
    def _forward_to_pooled_hidden(self, image):
        """
        Forward to the layer that is shared by all n_sources.

        Args:
            image:

        Returns:
        """

        log_img = torch.log(image - image.min() + 1.0)
        h = self.enc_conv(log_img)

        return self.enc_fc(h)

    def _get_var_params_all(self, image_patches):
        """
        Concatenate all output parameters for all possible n_sources
        Args:
            image_patches: A tensor of shape (n_patches, n_bands, patch_slen, patch_slen)

        Returns:
        """
        h = self._forward_to_pooled_hidden(image_patches)
        return self.enc_final(h)

    ######################
    # Forward modules
    ######################
    def forward(self, image_patches, n_sources=None):
        # pass through neural network, h is the array fo variational distribution parameters.
        # h has shape:
        h = self._get_var_params_all(image_patches)

        # get probability of n_sources
        log_probs_n = self._get_logprob_n_from_var_params(h)

        if n_sources is None:
            n_sources = torch.argmax(log_probs_n, dim=1)

        # extract parameters
        (
            loc_mean,
            loc_logvar,
            source_param_mean,
            source_param_logvar,
        ) = self._get_var_params_for_n_sources(
            h, n_sources=n_sources.clamp(max=self.max_detections)
        )

        # in the case of stars these are log_flux_mean, and log_flux_log_var.
        # loc_mean has shape = (n_patches x max_detections x len(x,y))
        return loc_mean, loc_logvar, source_param_mean, source_param_logvar, log_probs_n

    def _get_logprob_n_from_var_params(self, h):
        """
        Obtain log probability of number of n_sources.

        * Example: If max_detections = 3, then Tensor will be (num_patches x 3) since will return probability of
        having 0,1,2 stars.

        Args:
            h:

        Returns:
        """

        free_probs = h[:, self.prob_indx]
        return self.log_softmax(free_probs)

    def _get_var_params_for_n_sources(self, h, n_sources):
        """
        Index into all possible combinations of variational parameters (h) to obtain actually variational parameters
        for n_sources.
        Args:
            h: Huge triangular array with variational parameters.
            n_sources:

        Returns:

        """

        if len(n_sources.shape) == 1:
            n_sources = n_sources.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # this class takes in an array of n_stars, n_samples x batchsize
        assert h.shape[1] == self.dim_out_all
        assert h.shape[0] == n_sources.shape[1]

        n_samples = n_sources.shape[0]

        batchsize = h.size(0)
        _h = torch.cat((h, torch.zeros(batchsize, 1).to(const.device)), dim=1)

        loc_logit_mean = torch.gather(
            _h,
            1,
            self.locs_mean_indx_mat[n_sources.transpose(0, 1)].reshape(batchsize, -1),
        )
        loc_logvar = torch.gather(
            _h,
            1,
            self.locs_var_indx_mat[n_sources.transpose(0, 1)].reshape(batchsize, -1),
        )

        source_param_mean = torch.gather(
            _h,
            1,
            self.source_params_mean_indx_mat[n_sources.transpose(0, 1)].reshape(
                batchsize, -1
            ),
        )
        source_param_logvar = torch.gather(
            _h,
            1,
            self.source_params_var_indx_mat[n_sources.transpose(0, 1)].reshape(
                batchsize, -1
            ),
        )

        # reshape
        loc_logit_mean = loc_logit_mean.reshape(
            batchsize, n_samples, self.max_detections, 2
        ).transpose(0, 1)
        loc_logvar = loc_logvar.reshape(
            batchsize, n_samples, self.max_detections, 2
        ).transpose(0, 1)
        source_param_mean = source_param_mean.reshape(
            batchsize, n_samples, self.max_detections, self.n_source_params
        ).transpose(0, 1)
        source_param_logvar = source_param_logvar.reshape(
            batchsize, n_samples, self.max_detections, self.n_source_params
        ).transpose(0, 1)

        loc_mean = torch.sigmoid(loc_logit_mean) * (loc_logit_mean != 0).float()

        if squeeze_output:
            return (
                loc_mean.squeeze(0),
                loc_logvar.squeeze(0),
                source_param_mean.squeeze(0),
                source_param_logvar.squeeze(0),
            )
        else:
            return loc_mean, loc_logvar, source_param_mean, source_param_logvar

    def _get_hidden_indices(self):
        """
        Get indices necessary to maintain the huge triangular matrix array.
        Returns:

        """

        self.locs_mean_indx_mat = (
            torch.full(
                (self.max_detections + 1, 2 * self.max_detections), self.dim_out_all
            )
            .type(torch.LongTensor)
            .to(const.device)
        )

        self.locs_var_indx_mat = (
            torch.full(
                (self.max_detections + 1, 2 * self.max_detections), self.dim_out_all
            )
            .type(torch.LongTensor)
            .to(const.device)
        )

        self.source_params_mean_indx_mat = (
            torch.full(
                (self.max_detections + 1, self.n_source_params * self.max_detections),
                self.dim_out_all,
            )
            .type(torch.LongTensor)
            .to(const.device)
        )
        self.source_params_var_indx_mat = (
            torch.full(
                (self.max_detections + 1, self.n_source_params * self.max_detections),
                self.dim_out_all,
            )
            .type(torch.LongTensor)
            .to(const.device)
        )

        self.prob_indx = (
            torch.zeros(self.max_detections + 1).type(torch.LongTensor).to(const.device)
        )

        for n_detections in range(1, self.max_detections + 1):
            indx0 = (
                int(0.5 * n_detections * (n_detections - 1) * self.n_params_per_source)
                + (n_detections - 1)
                + 1
            )

            indx1 = (2 * n_detections) + indx0
            indx2 = (2 * n_detections) * 2 + indx0

            # indices for locations
            self.locs_mean_indx_mat[
                n_detections, 0 : (2 * n_detections)
            ] = torch.arange(indx0, indx1)
            self.locs_var_indx_mat[n_detections, 0 : (2 * n_detections)] = torch.arange(
                indx1, indx2
            )

            indx3 = indx2 + (n_detections * self.n_source_params)
            indx4 = indx3 + (n_detections * self.n_source_params)

            # indices for source params.
            self.source_params_mean_indx_mat[
                n_detections, 0 : (n_detections * self.n_source_params)
            ] = torch.arange(indx2, indx3)

            self.source_params_var_indx_mat[
                n_detections, 0 : (n_detections * self.n_source_params)
            ] = torch.arange(indx3, indx4)

            self.prob_indx[n_detections] = indx4

    ######################
    # Modules for patching images and parameters
    ######################
    def get_image_patches(
        self, images, locs=None, source_params=None, clip_max_sources=False
    ):
        """

        :param images: torch.Tensor of shape (batchsize x num_bands x slen x slen)
        :param locs:
        :param source_params: fluxes/log_fluxes/gal_params.
        :param clip_max_sources:
        :return:
        """
        assert len(images.shape) == 4  # should be batchsize x n_bands x slen x slen
        assert images.shape[1] == self.n_bands

        slen = images.shape[-1]

        # in case the image that we pass in to the encoder is of different size than the original
        # encoder should be able to handle these cases to.
        if not (images.shape[-1] == self.slen):
            # get the coordinates
            tile_coords = image_utils.get_tile_coords(
                slen, slen, self.patch_slen, self.step
            )
        else:
            # else, use the cached coordinates
            tile_coords = self.tile_coords

        image_patches = image_utils.tile_images(images, self.patch_slen, self.step)

        if (locs is not None) and (source_params is not None):
            assert source_params.shape[2] == self.n_source_params

            # get parameters in patch as well
            (
                patch_locs,
                patch_source_params,
                patch_n_sources,
                patch_is_on_array,
            ) = image_utils.get_params_in_patches(
                tile_coords,
                locs,
                source_params,
                slen,
                self.patch_slen,
                self.edge_padding,
            )

            if clip_max_sources:
                patch_n_sources = patch_n_sources.clamp(max=self.max_detections)
                patch_locs = patch_locs[:, 0 : self.max_detections, :]
                patch_source_params = patch_source_params[:, 0 : self.max_detections, :]
                patch_is_on_array = patch_is_on_array[:, 0 : self.max_detections]

        else:
            patch_locs = None
            patch_source_params = None
            patch_n_sources = None
            patch_is_on_array = None

        return (
            image_patches,
            patch_locs,
            patch_source_params,
            patch_n_sources,
            patch_is_on_array,
        )

    ######################
    # Modules to sample our variational distribution and get parameters on the full image
    ######################
    def _get_full_params_from_sampled_params(
        self, patch_locs_sampled, patch_source_params_sampled, slen
    ):

        n_samples = patch_locs_sampled.shape[0]
        n_image_patches = patch_locs_sampled.shape[1]

        assert self.n_source_params == patch_source_params_sampled.shape[-1]

        # if the image given is not the same as the original encoder training images.
        if not (slen == self.slen):
            tile_coords = image_utils.get_tile_coords(
                slen, slen, self.patch_slen, self.step
            )
        else:
            tile_coords = self.tile_coords

        assert (n_image_patches % tile_coords.shape[0]) == 0

        locs, source_params, n_sources = image_utils.get_full_params_from_patch_params(
            patch_locs_sampled.reshape(
                n_samples * n_image_patches, -1, 2
            ),  # 2 = len((x,y))
            patch_source_params_sampled.reshape(
                n_samples * n_image_patches, -1, self.n_source_params
            ),
            tile_coords,
            slen,
            self.patch_slen,
            self.edge_padding,
        )

        return locs, source_params, n_sources

    def _sample_patch_params(
        self,
        image,
        n_samples,
        return_map_n_sources,
        return_map_source_params,
        patch_n_sources,
        training,
    ):
        """
        NOTE: In the case of stars this will return log_fluxes!

        Args:
            image:
            n_samples:
            return_map_n_sources:
            return_map_source_params:
            patch_n_sources:
            training:

        Returns:

        """

        # our sampling only works for one image at a time at the moment ...
        assert image.shape[0] == 1

        image_patches = self.get_image_patches(image, locs=None, source_params=None)[0]

        # pass through NN
        h = self._get_var_params_all(image_patches)

        # get log probs for number of sources
        log_probs_n_source_patch = self._get_logprob_n_from_var_params(h)

        if not training:
            h = h.detach()
            log_probs_n_source_patch = log_probs_n_source_patch.detach()

        # sample number of stars
        if patch_n_sources is None:
            if return_map_n_sources:
                patch_n_stars_sampled = (
                    torch.argmax(log_probs_n_source_patch.detach(), dim=1)
                    .repeat(n_samples)
                    .view(n_samples, -1)
                )

            else:
                patch_n_stars_sampled = const.sample_class_weights(
                    torch.exp(log_probs_n_source_patch.detach()), n_samples
                ).view(n_samples, -1)
        else:
            patch_n_stars_sampled = patch_n_sources.repeat(n_samples).view(
                n_samples, -1
            )

        is_on_array = const.get_is_on_from_patch_n_sources_2d(
            patch_n_stars_sampled, self.max_detections
        )
        is_on_array = is_on_array.unsqueeze(3).float()

        # get variational parameters: these are on image patches
        (
            loc_mean,
            loc_logvar,
            source_param_mean,
            source_param_logvar,
        ) = self._get_var_params_for_n_sources(h, patch_n_stars_sampled)

        if return_map_source_params:
            loc_sd = const.FloatTensor(*loc_logvar.shape).zero_()
            source_params_sd = const.FloatTensor(*source_param_logvar.shape).zero_()
        else:
            loc_sd = torch.exp(0.5 * loc_logvar)
            source_params_sd = torch.exp(0.5 * source_param_logvar).clamp(max=0.5)

        # sample locations
        _locs_randn = const.FloatTensor(*loc_mean.shape).normal_()
        patch_locs_sampled = (loc_mean + _locs_randn * loc_sd) * is_on_array

        # sample source params, these are log_fluxes or latent galaxy params (normal variables)
        _source_params_randn = const.FloatTensor(*source_param_mean.shape).normal_()

        patch_source_params_sampled = (
            source_param_mean + _source_params_randn * source_params_sd
        )

        return patch_locs_sampled, patch_source_params_sampled, is_on_array

    def sample_encoder(
        self,
        image,
        n_samples=1,
        return_map_n_sources=False,
        return_map_source_params=False,
        patch_n_sources=None,
        training=False,
    ):
        """
        In the case of stars, this function will return log_fluxes as source_params. Can then obtain fluxes with the
        following procedure:

        >> is_on_array = const.get_is_on_from_n_stars(n_stars, max_stars)
        >> fluxes = np.exp(log_fluxes) * is_on_array

        where `max_stars` will correspond to the maximum number of stars that was used when simulating the `image`
        passed in to this function.

        Args:
            image:
            n_samples:
            return_map_n_sources:
            return_map_source_params:
            patch_n_sources:
            training:

        Returns:

        """
        slen = image.shape[-1]
        (
            patch_locs_sampled,
            patch_source_params_sampled,
            is_on_array,
        ) = self._sample_patch_params(
            image,
            n_samples,
            return_map_n_sources,
            return_map_source_params,
            patch_n_sources,
            training,
        )
        patch_source_params_sampled = patch_source_params_sampled * is_on_array

        # get parameters on full image
        locs, source_params, n_sources = self._get_full_params_from_sampled_params(
            patch_locs_sampled, patch_source_params_sampled, slen
        )

        # returns either galaxy_params or log_fluxes.
        return locs, source_params, n_sources
