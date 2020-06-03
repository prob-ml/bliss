import torch
import torch.nn as nn
from torch.distributions import categorical
from ..datasets.simulated_datasets import get_is_on_from_n_sources
from .. import device


def _sample_class_weights(class_weights, n_samples=1):
    """
    Draw a sample from Categorical variable with
    probabilities class_weights.
    """

    assert not torch.any(torch.isnan(class_weights))
    cat_rv = categorical.Categorical(probs=class_weights)
    return cat_rv.sample((n_samples,)).detach().squeeze()


def _extract_ptiles_2d(img, tile_shape, step=None, batch_first=False):
    """
    Take in an image (tensor) and the shape of the padded tile
    we want to separate it into and
    return the padded tiles also as a tensor.

    Taken from: https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78
    """
    if step is None:
        step = [1.0, 1.0]

    tile_H, tile_W = tile_shape[0], tile_shape[1]
    if img.size(2) < tile_H:
        num_padded_H_Top = (tile_H - img.size(2)) // 2
        num_padded_H_Bottom = tile_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if img.size(3) < tile_W:
        num_padded_W_Left = (tile_W - img.size(3)) // 2
        num_padded_W_Right = tile_W - img.size(3) - num_padded_W_Left
        padding_W = torch.nn.ConstantPad2d(
            (num_padded_W_Left, num_padded_W_Right, 0, 0), 0
        )
        img = padding_W(img)
    step_int = [0, 0]
    step_int[0] = int(tile_H * step[0]) if (isinstance(step[0], float)) else step[0]
    step_int[1] = int(tile_W * step[1]) if (isinstance(step[1], float)) else step[1]
    ptiles_fold_H = img.unfold(2, tile_H, step_int[0])
    if (img.size(2) - tile_H) % step_int[0] != 0:
        ptiles_fold_H = torch.cat(
            (ptiles_fold_H, img[:, :, -tile_H:,].permute(0, 1, 3, 2).unsqueeze(2)),
            dim=2,
        )
    ptiles_fold_HW = ptiles_fold_H.unfold(3, tile_W, step_int[1])
    if (img.size(3) - tile_W) % step_int[1] != 0:
        ptiles_fold_HW = torch.cat(
            (
                ptiles_fold_HW,
                ptiles_fold_H[:, :, :, -tile_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3),
            ),
            dim=3,
        )
    ptiles = ptiles_fold_HW.permute(2, 3, 0, 1, 4, 5)
    ptiles = ptiles.reshape(-1, img.size(0), img.size(1), tile_H, tile_W)
    if batch_first:
        ptiles = ptiles.permute(1, 0, 2, 3, 4)
    return ptiles


def _tile_images(images, ptile_slen, step):
    """
    Breaks up a large image into smaller padded tiles.
    Each tile has size ptile_slen x ptile_slen, where
    the number of padded tiles per image  is (slen - ptile_slen / step)**2.

    NOTE: input and output are torch tensors, not numpy arrays.

    :param images: A tensor of size (batchsize x n_bands x slen x slen)
    :param ptile_slen: The side length of each padded tile.
    :param step:
    :return: image_ptiles, output tensor of shape:
             (batchsize * ptiles per image) x n_bands x ptile_slen x ptile_slen
    :rtype: class:`torch.Tensor`
    """

    assert len(images.shape) == 4

    image_xlen = images.shape[2]
    image_ylen = images.shape[3]

    # My tile coords doesn't work otherwise ...
    assert (image_xlen - ptile_slen) % step == 0
    assert (image_ylen - ptile_slen) % step == 0

    n_bands = images.shape[1]
    image_ptiles = torch.tensor([], device=device)
    for b in range(n_bands):
        image_ptiles_b = _extract_ptiles_2d(
            images[:, b : (b + 1), :, :],
            tile_shape=[ptile_slen, ptile_slen],
            step=[step, step],
            batch_first=True,
        ).reshape(-1, 1, ptile_slen, ptile_slen)

        # torch.cat(...) works with empty tensors.
        image_ptiles = torch.cat((image_ptiles, image_ptiles_b), dim=1)

    return image_ptiles


def _get_tile_coords(image_xlen, image_ylen, ptile_slen, step):
    """
    This records (x0, x1) indices each image padded tile comes from.

    :param image_xlen: The x side length of the image in pixels.
    :param image_ylen: The y side length of the image in pixels.
    :param ptile_slen: The side length of the padded tile in pixels.
    :param step: pixels by which to shift every padded tile.
    :return: tile_coords, a torch.LongTensor
    """

    nx_ptiles = ((image_xlen - ptile_slen) // step) + 1
    ny_ptiles = ((image_ylen - ptile_slen) // step) + 1
    n_ptiles = nx_ptiles * ny_ptiles

    def return_coords(i):
        return [(i // ny_ptiles) * step, (i % ny_ptiles) * step]

    tile_coords = torch.tensor([return_coords(i) for i in range(n_ptiles)])
    tile_coords = tile_coords.long().to(device)

    return tile_coords


def _bring_to_front(
    n_sources,
    is_on_array,
    locs,
    galaxy_params,
    log_fluxes,
    galaxy_bool,
    n_galaxy_params,
    n_star_params,
):
    # puts all the on sources in front, returned dimension is max(n_sources)
    is_on_array_full = get_is_on_from_n_sources(n_sources, n_sources.max())
    indx = is_on_array_full.clone().long()
    indx[indx == 1] = torch.nonzero(is_on_array, as_tuple=False)[:, 1]

    new_galaxy_params = torch.gather(
        galaxy_params, dim=1, index=indx.unsqueeze(2).repeat(1, 1, n_galaxy_params)
    ) * is_on_array_full.float().unsqueeze(2)

    new_log_fluxes = torch.gather(
        log_fluxes, dim=1, index=indx.unsqueeze(2).repeat(1, 1, n_star_params)
    ) * is_on_array_full.float().unsqueeze(2)

    new_locs = torch.gather(
        locs, dim=1, index=indx.unsqueeze(2).repeat(1, 1, 2)
    ) * is_on_array_full.float().unsqueeze(2)

    new_galaxy_bool = torch.gather(
        galaxy_bool, dim=1, index=indx.unsqueeze(2).repeat(1, 1, 1)
    ) * is_on_array_full.float().unsqueeze(2)

    return (
        new_locs,
        new_galaxy_params,
        new_log_fluxes,
        new_galaxy_bool,
        is_on_array_full,
    )


# TODO: construct tile_* variables so we don't need to use this ugly function.
#       just switching out max_detections in _bring_to_front won't work, clipping the is_on_array
#       won't work because it might have sources in the last position. Need to sort first in a more
#       clever way.
def _apply_padding_and_clipping(
    max_detections,
    tile_n_sources,
    tile_locs,
    tile_galaxy_params,
    tile_log_fluxes,
    tile_galaxy_bool,
    tile_is_on_array,
):
    # In the loss function, it assumes that the true max number of stars on each tile
    # equals the max number of stars specified in the init of the encoder. Sometimes the
    # true max stars on tiles is less than the user-specified max stars, and this would
    # throw the error in the loss function. Padding solves this issue.

    # max number of stars seen in the each tile.
    max_n_stars_seen = tile_locs.size(1)
    if max_n_stars_seen < max_detections:
        n_pad = max_detections - tile_locs.size(1)
        n_tiles = tile_locs.size(0)

        assert tile_galaxy_params.size(0) == n_tiles
        assert tile_log_fluxes.size(0) == n_tiles
        assert tile_is_on_array.size(0) == n_tiles

        pad_zeros = torch.zeros(n_tiles, n_pad, tile_locs.size(-1), device=device,)
        tile_locs = torch.cat((tile_locs, pad_zeros), dim=1)

        pad_zeros2 = torch.zeros(
            n_tiles, n_pad, tile_galaxy_params.size(-1), device=device,
        )
        tile_galaxy_params = torch.cat((tile_galaxy_params, pad_zeros2), dim=1)

        pad_zeros3 = torch.zeros(
            n_tiles, n_pad, tile_log_fluxes.size(-1), device=device,
        )
        tile_log_fluxes = torch.cat((tile_log_fluxes, pad_zeros3), dim=1)

        pad_zeros4 = torch.zeros(n_tiles, n_pad, dtype=torch.long, device=device)
        tile_is_on_array = torch.cat((tile_is_on_array, pad_zeros4), dim=1)

        pad_zeros5 = torch.zeros(n_tiles, n_pad, dtype=torch.long, device=device)
        tile_galaxy_bool = torch.cat((tile_galaxy_bool, pad_zeros5), dim=1)

    # always clip max sources since it doesn't hurt.
    tile_n_sources = tile_n_sources.clamp(max=max_detections)
    tile_locs = tile_locs[:, 0:max_detections, :]
    tile_galaxy_params = tile_galaxy_params[:, 0:max_detections, :]
    tile_log_fluxes = tile_log_fluxes[:, 0:max_detections, :]
    tile_is_on_array = tile_is_on_array[:, 0:max_detections]
    tile_galaxy_bool = tile_galaxy_bool[:, 0:max_detections]

    return (
        tile_n_sources,
        tile_locs,
        tile_galaxy_params,
        tile_log_fluxes,
        tile_galaxy_bool,
        tile_is_on_array,
    )


def _get_full_params_from_tile_params(
    tile_coords,
    tile_locs,
    tile_galaxy_params,
    tile_log_fluxes,
    tile_galaxy_bool,
    full_slen,
    stamp_slen,
    edge_padding,
):
    # NOTE: off sources should have tile_locs == 0 and tile_galaxy_params == 0
    batchsize = int(tile_galaxy_params.shape[0] / tile_coords.shape[0])
    n_sources_in_batch = int(
        tile_galaxy_params.shape[0] * tile_galaxy_params.shape[1] / batchsize
    )

    assert tile_galaxy_params.size(0) == tile_log_fluxes.size(0)
    assert tile_galaxy_params.size(1) == tile_log_fluxes.size(1)
    assert (tile_galaxy_params.shape[0] % tile_coords.shape[0]) == 0
    assert (tile_log_fluxes.shape[0] % tile_coords.shape[0]) == 0
    assert (tile_galaxy_params.shape[0] % batchsize) == 0

    galaxy_params = tile_galaxy_params.view(batchsize, n_sources_in_batch, -1)
    log_fluxes = tile_log_fluxes.view(batchsize, n_sources_in_batch, -1)
    galaxy_bool = tile_galaxy_bool.view(batchsize, n_sources_in_batch, -1)

    scale = stamp_slen - 2 * edge_padding
    bias = tile_coords.repeat(batchsize, 1).unsqueeze(1).float() + edge_padding - 0.5
    locs = (tile_locs * scale + bias) / (full_slen - 1)
    locs = locs.view(batchsize, n_sources_in_batch, 2)

    tile_is_on_bool = (galaxy_params > 0).any(2).float()
    n_sources = torch.sum(tile_is_on_bool > 0, dim=1)

    (locs, galaxy_params, log_fluxes, galaxy_bool, _) = _bring_to_front(
        n_sources,
        tile_is_on_bool,
        locs,
        galaxy_params,
        log_fluxes,
        tile_galaxy_bool,
        galaxy_params.size(-1),
        log_fluxes.size(-1),
    )
    return locs, galaxy_params, log_fluxes, galaxy_bool, n_sources


class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)


class SourceEncoder(nn.Module):
    def __init__(
        self,
        slen,
        ptile_slen,
        step,
        edge_padding,
        n_bands,
        max_detections,
        n_star_params,
        n_galaxy_params,
        enc_conv_c=20,
        enc_kern=3,
        enc_hidden=256,
    ):
        """
        This class implements the source encoder, which is supposed to take in a synthetic image of
        size slen * slen
        and returns a NN latent variable representation of this image.

        * NOTE: Assumes that `source_params` are always `log_fluxes` throughout the code.

        * NOTE: Should have (n_bands == n_source_params) in the case of stars.

        * EXAMPLE on padding: If the ptile_slen=8, edge_padding=3, then the size of a tile will be
        8-2*3=2

        :param slen: dimension of full image, we assume its square for now
        :param ptile_slen: dimension (in pixels) of the individual
                           image padded tiles (usually 8 for stars, and _ for galaxies).
        :param step: number of pixels to shift every padded tile.
        :param edge_padding: length of padding (in pixels).
        :param n_bands : number of bands
        :param max_detections:
        * For fluxes this should equal number of bands, for galaxies it will be the number of latent
        dimensions in the network.
        """
        super(SourceEncoder, self).__init__()

        # image parameters
        self.slen = slen
        self.ptile_slen = ptile_slen
        self.step = step
        self.n_bands = n_bands

        self.edge_padding = edge_padding

        self.tile_coords = _get_tile_coords(slen, slen, self.ptile_slen, self.step)
        self.n_tiles = self.tile_coords.size(0)

        # max number of detections
        self.max_detections = max_detections

        # convolutional NN parameters
        self.enc_conv_c = enc_conv_c
        self.enc_kern = enc_kern
        self.enc_hidden = enc_hidden

        momentum = 0.5

        # convolutional NN
        self.enc_conv = nn.Sequential(
            nn.Conv2d(
                self.n_bands, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.enc_conv_c, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.BatchNorm2d(
                self.enc_conv_c, momentum=momentum, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.enc_conv_c, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.enc_conv_c, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.BatchNorm2d(
                self.enc_conv_c, momentum=momentum, track_running_stats=True
            ),
            nn.ReLU(),
            Flatten(),
        )

        # output dimension of convolutions
        conv_out_dim = self.enc_conv(
            torch.zeros(1, n_bands, ptile_slen, ptile_slen)
        ).size(1)

        # fully connected layers
        self.enc_fc = nn.Sequential(
            nn.Linear(conv_out_dim, self.enc_hidden),
            nn.BatchNorm1d(
                self.enc_hidden, momentum=momentum, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Linear(self.enc_hidden, self.enc_hidden),
            nn.BatchNorm1d(
                self.enc_hidden, momentum=momentum, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Linear(self.enc_hidden, self.enc_hidden),
            nn.BatchNorm1d(
                self.enc_hidden, momentum=momentum, track_running_stats=True
            ),
            nn.ReLU(),
        )

        # There are self.max_detections * (self.max_detections + 1)
        #  total possible detections, and each detection has
        #  4 + 2*n parameters (2 means and 2 variances for each loc + mean and variance for
        #  n source_param's (flux per band or galaxy params.) + 1 for the Bernoulli variable
        #  of whether the source is a star or galaxy.
        self.n_star_params = n_star_params
        self.n_galaxy_params = n_galaxy_params
        self.n_params_per_source = (
            4 + 2 * (self.n_star_params + self.n_galaxy_params) + 1
        )

        # The first term corresponds to: for each param, for each possible number of detection d,
        # there are d ways of assigning that param.
        # The second and third term accounts for categorical probability over # of objects.
        # These dimensions correspond to the probabilities in ONE tile.
        self.dim_out_all = int(
            0.5
            * self.max_detections
            * (self.max_detections + 1)
            * self.n_params_per_source
            + 1
            + self.max_detections
        )

        self.variational_params = [
            ("loc_mean", 2, lambda x: torch.sigmoid(x) * (x != 0).float()),
            ("loc_logvar", 2),
            ("log_fluxes_mean", self.n_star_params),
            ("log_fluxes_var", self.n_star_params),
            ("galaxy_params_mean", self.n_galaxy_params),
            ("galaxy_params_var", self.n_galaxy_params),
            ("prob_galaxy", 1, lambda x: torch.sigmoid(x)),
        ]
        self.n_variational_params = len(self.variational_params)

        self.indx_mats, self.prob_n_source_indx = self._get_hidden_indices()

        self.enc_final = nn.Linear(self.enc_hidden, self.dim_out_all)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _create_indx_mats(self):
        indx_mats = []
        for i in range(self.n_variational_params):
            param_dim = self.variational_params[i][1]
            shape = (self.max_detections + 1, param_dim * self.max_detections)
            indx_mat = torch.full(
                shape, self.dim_out_all, dtype=torch.long, device=device,
            )
            indx_mats.append(indx_mat)
        return indx_mats

    def _update_indx_mat_for_n_detections(self, indx_mats, curr_indx, n_detections):
        # add corresponding indices to index matrices for n_detections.
        for i in range(self.n_variational_params):
            param_dim = self.variational_params[i][1]
            indx_mat = indx_mats[i]
            new_indx = (param_dim * n_detections) + curr_indx
            indx_mat[n_detections, 0 : (param_dim * n_detections)] = torch.arange(
                curr_indx, new_indx
            )
            indx_mats[i] = indx_mat
            curr_indx = new_indx

        return indx_mats, curr_indx

    def _get_hidden_indices(self):
        """Setup the indices corresponding to entries in h, these are cached since same for all h.
        """

        indx_mats = self._create_indx_mats()  # same order as self.variational_params
        prob_n_source_indx = torch.zeros(
            self.max_detections + 1, dtype=torch.long, device=device
        )

        for n_detections in range(1, self.max_detections + 1):
            # index corresponding to where we left off in last iteration.
            curr_indx = (
                int(0.5 * n_detections * (n_detections - 1) * self.n_params_per_source)
                + (n_detections - 1)
                + 1
            )

            # add corresponding indices to the index matrices of variational params.
            indx_mats, curr_indx = self._update_indx_mat_for_n_detections(
                indx_mats, curr_indx, n_detections
            )

            # the categorical prob will go at the end of the rest.
            prob_n_source_indx[n_detections] = curr_indx

        return indx_mats, prob_n_source_indx

    ######################
    # Forward modules
    ######################

    def _indx_h_for_n_sources(self, h, n_sources, indx_mat, param_dim):
        """
        Index into all possible combinations of variational parameters (h) to obtain actually
        variational parameters for n_sources.
        Args:
            h: shape = (n_ptiles x dim_out_all)
            n_sources: n_samples x n_tiles
            param_dim: the dimension of the parameter you are indexing h for. e.g. for locs,
                            dim_per_source = 2, for galaxy params we usually have
                            dim_per_source = 8.
        Returns:
            var_param: shape = (n_samples x n_ptiles x max_detections x dim_per_source)
        """

        assert len(n_sources.shape) == 2, "Shape: (n_samples x n_ptiles)"
        assert h.size(0) == n_sources.size(1)  # = n_ptiles
        assert h.size(1) == self.dim_out_all

        n_ptiles = h.size(0)
        n_samples = n_sources.size(0)

        # append null column, return zero if indx_mat returns null index (dim_out_all)
        _h = torch.cat((h, torch.zeros(n_ptiles, 1, device=device)), dim=1)

        # select the indices from _h indicated by indx_mat.
        var_param = torch.gather(
            _h, 1, indx_mat[n_sources.transpose(0, 1)].reshape(n_ptiles, -1),
        )

        var_param = var_param.reshape(
            n_ptiles, n_samples, self.max_detections, param_dim
        ).transpose(0, 1)

        # shape = (n_samples x n_ptiles x max_detections x dim_per_source)
        return var_param

    def _get_logprob_n_from_var_params(self, h):
        """
        Obtain log probability of number of n_sources.

        * Example: If max_detections = 3, then Tensor will be (n_tiles x 3) since will return
        probability of having 0,1,2 stars.
        """
        free_probs = h[:, self.prob_n_source_indx]
        return self.log_softmax(free_probs)

    def _get_var_params_for_n_sources(self, h, n_sources):
        """
        Returns:
            loc_mean.shape = (n_samples x n_ptiles x max_detections x len(x,y))
            source_param_mean.shape = (n_samples x n_ptiles x max_detections x n_source_params)
        """

        estimated_params = []
        for i in range(self.n_variational_params):
            indx_mat = self.indx_mats[i]
            param_info = self.variational_params[i]
            param_dim = param_info[1]

            # obtain hidden function to apply if included, otherwise do nothing.
            hidden_function = param_info[2] if len(param_info) > 2 else lambda x: x
            _param = self._indx_h_for_n_sources(h, n_sources, indx_mat, param_dim)
            param = hidden_function(_param)
            estimated_params.append(param)

        return estimated_params

    def _get_var_params_all(self, image_ptiles):
        """get h matrix.

        image_ptiles shape: (n_ptiles, n_bands, ptile_slen, ptile_slen)
        """
        # Forward to the layer that is shared by all n_sources.
        log_img = torch.log(image_ptiles - image_ptiles.min() + 1.0)
        h = self.enc_conv(log_img)
        h = self.enc_fc(h)

        # Concatenate all output parameters for all possible n_sources
        return self.enc_final(h)

    def forward(self, image_ptiles, n_sources):
        # will unsqueeze and squeeze n_sources later.
        assert len(n_sources.shape) == 1
        n_sources = n_sources.unsqueeze(0)

        # h.shape = (n_ptiles x self.dim_out_all)
        h = self._get_var_params_all(image_ptiles)

        # get probability of n_sources
        # shape = (n_ptiles x (max_detections+1))
        log_probs_n_sources = self._get_logprob_n_from_var_params(h)

        # extract parameters
        prob_galaxy = self._get_prob_galaxy_for_n_sources(
            h, n_sources=n_sources.clamp(max=self.max_detections)
        )

        # loc_mean has shape = (1 x n_ptiles x max_detections x len(x,y))
        (
            loc_mean,
            loc_logvar,
            log_flux_mean,
            log_flux_logvar,
            galaxy_param_mean,
            galaxy_param_logvar,
        ) = self._get_var_params_for_n_sources(
            h, n_sources=n_sources.clamp(max=self.max_detections)
        )

        # in the case of stars these are log_flux_mean, and log_flux_logvar.
        return (
            loc_mean.squeeze(0),
            loc_logvar.squeeze(0),
            log_flux_mean.squeeze(0),
            log_flux_logvar.squeeze(0),
            galaxy_param_mean.squeeze(0),
            galaxy_param_logvar.squeeze(0),
            prob_galaxy.squeeze(0),
            log_probs_n_sources.squeeze(0),
        )

    def _get_tile_coords(self, slen):
        tile_coords = self.tile_coords

        # handle cases where images passed in are not of original size.
        if not (slen == self.slen):
            tile_coords = _get_tile_coords(slen, slen, self.ptile_slen, self.step)
        return tile_coords

    def get_image_ptiles(self, images):
        assert len(images.shape) == 4  # should be batchsize x n_bands x slen x slen
        assert images.size(1) == self.n_bands

        image_ptiles = _tile_images(images, self.ptile_slen, self.step)
        return image_ptiles

    def _get_locs_in_tiles(
        tile_coords,
        locs,
        galaxy_params,
        log_fluxes,
        galaxy_bool,
        slen,
        ptile_slen,
        edge_padding=0,
    ):
        # locs are the coordinates in the full image, in coordinates between 0-1
        assert torch.all(locs <= 1.0)
        assert torch.all(locs >= 0.0)

        n_ptiles = tile_coords.size(0)  # number of ptiles in a full image
        fullimage_batchsize = locs.size(0)  # number of full images
        max_sources = locs.size(1)
        subimage_batchsize = n_ptiles * fullimage_batchsize  # total number of ptiles

        tile_coords = tile_coords.unsqueeze(0).unsqueeze(2).float()
        locs = locs * (slen - 1)

        # indicator for each ptile, whether there is a loc there or not (loc order maintained)
        which_locs_array = (
            (locs.unsqueeze(1) > tile_coords + edge_padding - 0.5)
            & (locs.unsqueeze(1) < tile_coords - 0.5 + ptile_slen - edge_padding)
            & (locs.unsqueeze(1) != 0)
        )
        which_locs_array = which_locs_array[:, :, :, 0] * which_locs_array[:, :, :, 1]
        which_locs_array = which_locs_array.float()

        # for each tile returned re-normalized locs, maintaining relative ordering of locs
        # (including leading/trailing zeroes) in the case that there are multiple objects
        # in that tile.
        tile_locs = which_locs_array.unsqueeze(3) * locs.unsqueeze(1)
        tile_locs -= tile_coords + edge_padding - 0.5  # centering relative to each tile
        tile_locs = tile_locs.view(subimage_batchsize, max_sources, 2)
        tile_locs /= ptile_slen - 2 * edge_padding  # normalization in each tile.
        tile_locs = torch.relu(tile_locs)  # some are negative now; set these to 0

        # now for log_fluxes and galaxy_params
        assert fullimage_batchsize == log_fluxes.size(0) == galaxy_params.size(0)
        assert max_sources == log_fluxes.size(1) == galaxy_params.size(1)
        n_star_params = log_fluxes.size(-1)
        n_galaxy_params = galaxy_params.size(-1)

        tile_log_fluxes = (
            which_locs_array.unsqueeze(3) * log_fluxes.unsqueeze(1)
        ).view(subimage_batchsize, max_sources, n_star_params)

        tile_galaxy_params = (
            which_locs_array.unsqueeze(3) * galaxy_params.unsqueeze(1)
        ).view(subimage_batchsize, max_sources, n_galaxy_params)

        tile_galaxy_bool = (
            which_locs_array.unsqueeze(3) * galaxy_bool.unsqueeze(2)
        ).view(subimage_batchsize, max_sources, 1)

        # sort locs so all the zeros are at the end
        is_on_array = (
            which_locs_array.view(subimage_batchsize, max_sources).float().to(device)
        )
        n_sources_per_tile = is_on_array.float().sum(dim=1).float().to(device)
        (
            tile_locs,
            tile_galaxy_params,
            tile_log_fluxes,
            tile_galaxy_bool,
            tile_is_on_array,
        ) = _bring_to_front(
            n_sources_per_tile,
            is_on_array,
            tile_locs,
            tile_galaxy_params,
            tile_log_fluxes,
            tile_galaxy_bool,
            n_galaxy_params,
            n_star_params,
        )

        return (
            tile_locs,
            tile_galaxy_params,
            tile_log_fluxes,
            tile_galaxy_bool,
            n_sources_per_tile,
            tile_is_on_array,
        )

    # def get_tiled_params(self, slen, locs, *params):
    #     # returned params in tiles as organized by locs.
    #     tile_coords = self._get_tile_coords(slen)
    #
    #     # first obtain tiled locs and is_on_array, also index for how to order rest.
    #     tile_locs, tile_is_on_array, indx_sort = self.
    #
    #     # then the rest of
    #
    #
    #         # get parameters in tiles as well
    #         (
    #             tile_locs,
    #             tile_galaxy_params,
    #             tile_log_fluxes,
    #             tile_galaxy_bool,
    #             tile_n_sources,
    #             tile_is_on_array,
    #         ) = _get_params_in_tiles(
    #             tile_coords,
    #             locs,
    #             galaxy_params,
    #             log_fluxes,
    #             galaxy_bool,
    #             slen,
    #             self.ptile_slen,
    #             self.edge_padding,
    #         )
    #
    #         (
    #             tile_n_sources,
    #             tile_locs,
    #             tile_galaxy_params,
    #             tile_log_fluxes,
    #             tile_galaxy_bool,
    #             tile_is_on_array,
    #         ) = _apply_padding_and_clipping(
    #             self.max_detections,
    #             tile_n_sources,
    #             tile_locs,
    #             tile_galaxy_params,
    #             tile_log_fluxes,
    #             tile_galaxy_bool,
    #             tile_is_on_array,
    #         )

    ######################
    # Modules to sample our variational distribution and get parameters on the full image
    ######################
    def _get_full_params_from_sampled_params(
        self,
        tile_locs_sampled,
        tile_galaxy_params_sampled,
        tile_log_fluxes_sampled,
        tile_galaxy_bool_sampled,
        slen,
    ):

        n_samples = tile_locs_sampled.shape[0]
        n_image_ptiles = tile_locs_sampled.shape[1]

        assert self.n_galaxy_params == tile_galaxy_params_sampled.shape[-1]
        assert self.n_star_params == tile_log_fluxes_sampled.shape[-1]

        tile_coords = self.tile_coords
        if not (slen == self.slen):
            tile_coords = _get_ptile_coords(slen, slen, self.ptile_slen, self.step)

        assert (n_image_ptiles % tile_coords.shape[0]) == 0

        _tile_locs_sampled = tile_locs_sampled.reshape(
            n_samples * n_image_ptiles, -1, 2
        )
        _tile_galaxy_params_sampled = tile_galaxy_params_sampled.reshape(
            n_samples * n_image_ptiles, -1, self.n_galaxy_params
        )
        _tile_log_fluxes_sampled = tile_log_fluxes_sampled.reshape(
            n_samples * n_image_ptiles, -1, self.n_star_params
        )
        _tile_galaxy_bool_sampled = tile_galaxy_bool_sampled.reshape(
            n_samples * n_image_ptiles, -1, 1
        )
        (
            locs,
            galaxy_params,
            log_fluxes,
            galaxy_bool,
            n_sources,
        ) = _get_full_params_from_tile_params(
            tile_coords,
            _tile_locs_sampled,
            _tile_galaxy_params_sampled,
            _tile_log_fluxes_sampled,
            _tile_galaxy_bool_sampled,
            slen,
            self.ptile_slen,
            self.edge_padding,
        )

        return locs, galaxy_params, log_fluxes, galaxy_bool, n_sources

    def _sample_tile_params(
        self, image, n_samples, return_map_n_sources, return_map_source_params,
    ):

        assert (
            image.size(0) == 1
        ), "Sampling only works for one image at a time for now..."

        # shape = (n_ptiles x n_bands x ptile_slen x ptile_slen)
        image_ptiles = self.get_image_ptiles(image)[0]

        # pass through NN
        # shape = (n_ptiles x dim_out_all)
        h = self._get_var_params_all(image_ptiles)

        # get log probs for number of sources
        # shape = (n_ptiles x max_detections)
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(h)

        # sample number of stars
        # output shape = (n_samples x n_ptiles)
        if return_map_n_sources:
            tile_n_sources_sampled = (
                torch.argmax(log_probs_n_sources_per_tile.detach(), dim=1)
                .repeat(n_samples)
                .view(n_samples, -1)
            )

        else:
            tile_n_sources_sampled = _sample_class_weights(
                torch.exp(log_probs_n_sources_per_tile.detach()), n_samples
            ).view(n_samples, -1)

        is_on_array = get_is_on_from_n_sources(
            tile_n_sources_sampled, self.max_detections
        )
        # shape = (n_samples x n_ptiles x max_detections x 1 )
        is_on_array = is_on_array.unsqueeze(3).float()

        # get variational parameters: these are on image tiles
        # loc_mean.shape = (n_samples x n_ptiles x max_detections x len(x,y))
        (
            loc_mean,
            loc_logvar,
            galaxy_param_mean,
            galaxy_param_logvar,
            log_flux_mean,
            log_flux_logvar,
            prob_galaxy,
        ) = self._get_var_params_for_n_sources(h, tile_n_sources_sampled)

        #  TODO: finish adding bernoulli prediction.
        if return_map_source_params:
            # bernoulli_shape = torch.Size([tile_n_sources.size(0), self.max_detections])
            # bernoulli_input = torch.full(bernoulli_shape, )
            tile_galaxy_bool_sampled = torch.zeros()
            loc_sd = torch.zeros_like(loc_logvar)
            galaxy_param_sd = torch.zeros_like(galaxy_param_logvar)
            log_flux_sd = torch.zeros_like(log_flux_logvar)
        else:
            loc_sd = torch.exp(0.5 * loc_logvar)
            galaxy_param_sd = torch.exp(0.5 * galaxy_param_logvar).clamp(max=0.5)
            log_flux_sd = torch.exp(0.5 * log_flux_logvar).clamp(max=0.5)

        # shape = (n_samples x n_ptiles x max_detections x len(x,y))
        assert loc_mean.shape == loc_sd.shape, "Shapes need to match"
        assert galaxy_param_mean.shape == galaxy_param_sd.shape, "Shapes need to match"
        assert log_flux_mean.shape == log_flux_sd.shape, "Shapes need to match"

        tile_locs_sampled = torch.normal(loc_mean, loc_sd) * is_on_array

        tile_galaxy_params_sampled = torch.normal(galaxy_param_mean, galaxy_param_sd)
        tile_galaxy_params_sampled *= is_on_array

        tile_log_fluxes_sampled = torch.normal(log_flux_mean, log_flux_sd)
        tile_log_fluxes_sampled *= is_on_array

        return (
            tile_locs_sampled,
            tile_galaxy_params_sampled,
            tile_log_fluxes_sampled,
            tile_galaxy_bool_sampled,
            is_on_array,
        )

    def sample_encoder(
        self,
        image,
        n_samples=1,
        return_map_n_sources=False,
        return_map_source_params=False,
    ):
        """
        In the case of stars, this function will return log_fluxes as source_params. Can then obtain
        fluxes with the following procedure:

        >> is_on_array = get_is_on_from_n_stars(n_stars, max_stars)
        >> fluxes = np.exp(log_fluxes) * is_on_array

        where `max_stars` corresponds to the maximum number of stars in a scene that was used when
        simulating the `image` passed in to this function.

        """

        slen = image.shape[-1]
        (
            tile_locs_sampled,
            tile_galaxy_params_sampled,
            tile_log_fluxes_sampled,
            tile_galaxy_bool_sampled,
            is_on_array,
        ) = self._sample_tile_params(
            image, n_samples, return_map_n_sources, return_map_source_params,
        )

        # get parameters on full image
        (
            locs,
            galaxy_params,
            log_fluxes,
            n_sources,
        ) = self._get_full_params_from_sampled_params(
            tile_locs_sampled,
            tile_galaxy_params_sampled,
            tile_log_fluxes_sampled,
            tile_galaxy_bool_sampled,
            slen,
        )

        # returns either galaxy_params or log_fluxes.
        return locs, galaxy_params, log_fluxes, n_sources
