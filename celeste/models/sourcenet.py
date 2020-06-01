import numpy as np
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

    :param img:
    :type img: class: `torch.Tensor`
    :param tile_shape:
    :param step:
    :param batch_first:
    :return: A tensor of padded tiles.
    :rtype: class: `torch.Tensor`
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
    :type images: class:`torch.Tensor`
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
    image_ptiles = None
    for b in range(n_bands):
        image_ptiles_b = _extract_ptiles_2d(
            images[:, b : (b + 1), :, :],
            tile_shape=[ptile_slen, ptile_slen],
            step=[step, step],
            batch_first=True,
        ).reshape(-1, 1, ptile_slen, ptile_slen)

        if b == 0:
            image_ptiles = image_ptiles_b
        else:
            image_ptiles = torch.cat((image_ptiles, image_ptiles_b), dim=1)

    return image_ptiles


def _get_ptile_coords(image_xlen, image_ylen, ptile_slen, step):
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

    tile_coords = (
        torch.from_numpy(np.array([return_coords(i) for i in range(n_ptiles)]))
        .long()
        .to(device)
    )

    return tile_coords


def _bring_to_front(
    n_sources,
    is_on_array,
    locs,
    galaxy_params,
    log_fluxes,
    n_galaxy_params,
    n_star_params,
):
    # puts all the on sources in front, returned dimension is max(n_sources)
    is_on_array_full = get_is_on_from_n_sources(n_sources, n_sources.max())
    indx = is_on_array_full.clone()
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

    return new_locs, new_galaxy_params, new_log_fluxes, is_on_array_full


def _get_params_in_tiles(
    tile_coords, locs, galaxy_params, log_fluxes, slen, ptile_slen, edge_padding=0,
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

    # obtain indicator for each ptile, whether there is a loc there.
    which_locs_array = (
        (locs.unsqueeze(1) > tile_coords + edge_padding - 0.5)
        & (locs.unsqueeze(1) < tile_coords - 0.5 + ptile_slen - edge_padding)
        & (locs.unsqueeze(1) != 0)
    )
    which_locs_array = (
        which_locs_array[:, :, :, 0] * which_locs_array[:, :, :, 1]
    ).float()

    tile_locs = (
        which_locs_array.unsqueeze(3) * locs.unsqueeze(1)
        - (tile_coords + edge_padding - 0.5)
    ).view(subimage_batchsize, max_sources, 2) / (ptile_slen - 2 * edge_padding)
    tile_locs = torch.relu(
        tile_locs
    )  # by subtracting, some are negative now; just set these to 0

    # now for log_fluxes and galaxy_params
    assert fullimage_batchsize == log_fluxes.size(0) == galaxy_params.size(0)
    assert max_sources == log_fluxes.size(1) == galaxy_params.size(1)
    n_star_params = log_fluxes.size(-1)
    n_galaxy_params = galaxy_params.size(-1)
    tile_log_fluxes = (which_locs_array.unsqueeze(3) * log_fluxes.unsqueeze(1)).view(
        subimage_batchsize, max_sources, n_star_params
    )
    tile_galaxy_params = (
        which_locs_array.unsqueeze(3) * galaxy_params.unsqueeze(1)
    ).view(subimage_batchsize, max_sources, n_galaxy_params)

    # sort locs so all the zeros are at the end
    is_on_array = (
        which_locs_array.view(subimage_batchsize, max_sources).long().to(device)
    )
    n_sources_per_tile = is_on_array.float().sum(dim=1).long().to(device)
    tile_locs, tile_galaxy_params, tile_log_fluxes, tile_is_on_array = _bring_to_front(
        n_sources_per_tile,
        is_on_array,
        tile_locs,
        tile_galaxy_params,
        tile_log_fluxes,
        n_galaxy_params,
        n_star_params,
    )

    return (
        tile_locs,
        tile_galaxy_params,
        tile_log_fluxes,
        n_sources_per_tile,
        tile_is_on_array,
    )


# TODO: construct tile_* variables so we don't need to do this padding.
def _apply_padding_and_clipping(
    max_detections,
    tile_n_sources,
    tile_locs,
    tile_galaxy_params,
    tile_log_fluxes,
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

    # always clip max sources since it doesn't hurt.
    tile_n_sources = tile_n_sources.clamp(max=max_detections)
    tile_locs = tile_locs[:, 0:max_detections, :]
    tile_galaxy_params = tile_galaxy_params[:, 0:max_detections, :]
    tile_log_fluxes = tile_log_fluxes[:, 0:max_detections, :]
    tile_is_on_array = tile_is_on_array[:, 0:max_detections]

    return (
        tile_n_sources,
        tile_locs,
        tile_galaxy_params,
        tile_log_fluxes,
        tile_is_on_array,
    )


def get_is_on_from_tile_n_sources_2d(tile_n_sources, max_sources):
    """

    :param tile_n_sources: A tensor of shape (n_samples x n_tiles), indicating the number of sources
                            at sample i, batch j. (n_samples = batchsize)
    :type tile_n_sources: class: `torch.Tensor`
    :param max_sources:
    :type max_sources: int
    :return:
    """
    assert not torch.any(torch.isnan(tile_n_sources))
    assert torch.all(tile_n_sources >= 0)
    assert torch.all(tile_n_sources <= max_sources)

    n_samples = tile_n_sources.shape[0]
    batchsize = tile_n_sources.shape[1]

    is_on_array = torch.zeros(
        n_samples, batchsize, max_sources, device=device, dtype=torch.long
    )

    for i in range(max_sources):
        is_on_array[:, :, i] = tile_n_sources > i

    return is_on_array


def get_full_params_from_tile_params(
    tile_locs, tile_source_params, tile_coords, full_slen, stamp_slen, edge_padding
):
    # NOTE: off sources should have tile_locs == 0 and tile_source_params == 0

    # reshaped before passing in into shape (batchsize * n_image_ptiles, -1, self.n_source_params)
    assert (tile_source_params.shape[0] % tile_coords.shape[0]) == 0
    batchsize = int(tile_source_params.shape[0] / tile_coords.shape[0])

    assert (tile_source_params.shape[0] % batchsize) == 0
    n_sources_in_batch = int(
        tile_source_params.shape[0] * tile_source_params.shape[1] / batchsize
    )

    n_source_params = tile_source_params.shape[2]  # = n_bands in the case of fluxes.
    source_params = tile_source_params.view(
        batchsize, n_sources_in_batch, n_source_params
    )

    scale = stamp_slen - 2 * edge_padding
    bias = tile_coords.repeat(batchsize, 1).unsqueeze(1).float() + edge_padding - 0.5
    locs = (tile_locs * scale + bias) / (full_slen - 1)

    locs = locs.view(batchsize, n_sources_in_batch, 2)

    tile_is_on_bool = (
        (source_params > 0).any(2).float()
    )  # if source_param in any n_source_params is nonzero
    n_sources = torch.sum(tile_is_on_bool > 0, dim=1)

    # puts all the on sources in front (of each tile subarray)
    is_on_array_full = get_is_on_from_n_sources(n_sources, n_sources.max())
    indx = is_on_array_full.clone()
    indx[indx == 1] = torch.nonzero(tile_is_on_bool, as_tuple=False)[:, 1]

    source_params, locs, _ = _bring_to_front(
        n_source_params, n_sources, tile_is_on_bool, source_params, locs
    )
    return locs, source_params, n_sources


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

        self.tile_coords = _get_ptile_coords(
            self.slen, self.slen, self.ptile_slen, self.step
        )
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
        self._get_hidden_indices()

        self.enc_final = nn.Linear(self.enc_hidden, self.dim_out_all)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _create_indx_mat(self, variational_params):
        for param_name, param_dim in variational_params:
            shape = (self.max_detections + 1, param_dim * self.max_detections)
            indx_mat = torch.full(
                shape, self.dim_out_all, dtype=torch.long, device=device,
            )
            setattr(self, param_name, indx_mat)

    def _update_indx_mat_for_n_detections(
        self, n_detections, curr_indx, variational_params
    ):
        for param_name, param_dim in variational_params:
            indx_mat = getattr(self, param_name)
            new_indx = (param_dim * n_detections) + curr_indx
            indx_mat[n_detections, 0 : (param_dim * n_detections)] = torch.arange(
                curr_indx, new_indx
            )
            setattr(self, param_name, indx_mat)
            curr_indx = new_indx

        return curr_indx

    def _get_hidden_indices(self):
        """
        Setup the indices corresponding to entries in h, these are just cached since they are the
        same for all h.
        Returns:
        """

        variational_params = [
            ("locs_mean", 2),
            ("locs_var", 2),
            ("log_fluxes_mean", self.n_star_params),
            ("log_fluxes_var", self.n_star_params),
            ("galaxy_params_mean", self.n_galaxy_params),
            ("galaxy_params_var", self.n_galaxy_params),
            ("prob_galaxy", 1),
        ]
        variational_params = [(f"{x[0]}_indx_mat", x[1]) for x in variational_params]

        # create index matrices attribute for variational parameters.
        self._create_indx_mat(variational_params)
        self.prob_n_source_indx = torch.zeros(
            self.max_detections + 1, dtype=torch.long, device=device
        )

        for n_detections in range(1, self.max_detections + 1):
            # index corresponding to where we left off in last iteration.
            curr_indx = (
                int(0.5 * n_detections * (n_detections - 1) * self.n_params_per_source)
                + (n_detections - 1)
                + 1
            )

            curr_indx = self._update_indx_mat_for_n_detections(
                n_detections, curr_indx, variational_params
            )

            # the categorical prob will go at the end.
            self.prob_n_source_indx[n_detections] = curr_indx

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

        # append null column
        _h = torch.cat((h, torch.zeros(n_ptiles, 1, device=device)), dim=1)

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

    def _get_prob_galaxy_for_n_sources(self, h, n_sources):
        return torch.sigmoid(
            self._indx_h_for_n_sources(h, n_sources, self.prob_galaxy_indx_mat, 1)
        )

    def _get_var_params_for_n_sources(self, h, n_sources):
        """
        Returns:
            loc_mean.shape = (n_samples x n_ptiles x max_detections x len(x,y))
            source_param_mean.shape = (n_samples x n_ptiles x max_detections x n_source_params)
        """

        loc_logit_mean = self._indx_h_for_n_sources(
            h, n_sources, self.locs_mean_indx_mat, 2
        )
        loc_logvar = self._indx_h_for_n_sources(h, n_sources, self.locs_var_indx_mat, 2)

        log_flux_mean = self._indx_h_for_n_sources(
            h, n_sources, self.log_fluxes_mean_indx_mat, self.n_star_params
        )
        log_flux_logvar = self._indx_h_for_n_sources(
            h, n_sources, self.log_fluxes_var_indx_mat, self.n_star_params
        )

        galaxy_param_mean = self._indx_h_for_n_sources(
            h, n_sources, self.galaxy_params_mean_indx_mat, self.n_galaxy_params
        )
        galaxy_param_logvar = self._indx_h_for_n_sources(
            h, n_sources, self.galaxy_params_var_indx_mat, self.n_galaxy_params
        )

        loc_mean = torch.sigmoid(loc_logit_mean) * (loc_logit_mean != 0).float()

        return (
            loc_mean,
            loc_logvar,
            log_flux_mean,
            log_flux_logvar,
            galaxy_param_mean,
            galaxy_param_logvar,
        )

    ############################
    # The layers of our neural network
    ############################
    def _forward_to_pooled_hidden(self, image):
        """
        Forward to the layer that is shared by all n_sources.
        """

        log_img = torch.log(image - image.min() + 1.0)
        h = self.enc_conv(log_img)

        return self.enc_fc(h)

    def _get_var_params_all(self, image_ptiles):
        """
        Concatenate all output parameters for all possible n_sources
        Args:
            image_ptiles: A tensor of shape (n_ptiles, n_bands, ptile_slen, ptile_slen)
        """
        h = self._forward_to_pooled_hidden(image_ptiles)
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

    ######################
    # Modules for tiling images and parameters
    ######################
    def get_image_ptiles(
        self, images, locs=None, galaxy_params=None, log_fluxes=None, galaxy_bool=None,
    ):
        assert len(images.shape) == 4  # should be batchsize x n_bands x slen x slen
        assert images.size(1) == self.n_bands

        slen = images.size(-1)
        tile_coords = self.tile_coords

        # handle cases where images passed in are not of original size.
        if not (slen == self.slen):
            tile_coords = _get_ptile_coords(slen, slen, self.ptile_slen, self.step)

        image_ptiles = _tile_images(images, self.ptile_slen, self.step)

        if locs is not None:
            assert galaxy_params is not None
            assert galaxy_bool is not None
            assert galaxy_params.size(-1) == self.n_source_params
            assert log_fluxes is not None and log_fluxes.size(-1) == self.n_bands

            # get parameters in tiles as well
            (
                tile_locs,
                tile_galaxy_params,
                tile_log_fluxes,
                tile_n_sources,
                tile_is_on_array,
            ) = _get_params_in_tiles(
                tile_coords,
                locs,
                galaxy_params,
                log_fluxes,
                slen,
                self.ptile_slen,
                self.edge_padding,
            )

            (
                tile_n_sources,
                tile_locs,
                tile_galaxy_params,
                tile_log_fluxes,
                tile_is_on_array,
            ) = _apply_padding_and_clipping(
                self.max_detections,
                tile_n_sources,
                tile_locs,
                tile_galaxy_params,
                tile_log_fluxes,
                tile_is_on_array,
            )

        else:
            tile_locs = None
            tile_galaxy_params = None
            tile_log_fluxes = None
            tile_n_sources = None
            tile_is_on_array = None

        return (
            image_ptiles,
            tile_locs,
            tile_galaxy_params,
            tile_log_fluxes,
            tile_n_sources,
            tile_is_on_array,
        )

    ######################
    # Modules to sample our variational distribution and get parameters on the full image
    ######################
    def _get_full_params_from_sampled_params(
        self, tile_locs_sampled, tile_source_params_sampled, slen
    ):

        n_samples = tile_locs_sampled.shape[0]
        n_image_ptiles = tile_locs_sampled.shape[1]

        assert self.n_source_params == tile_source_params_sampled.shape[-1]

        # if the image given is not the same as the original encoder training images.
        if not (slen == self.slen):
            tile_coords = _get_ptile_coords(slen, slen, self.ptile_slen, self.step)
        else:
            tile_coords = self.tile_coords

        assert (n_image_ptiles % tile_coords.shape[0]) == 0

        (locs, source_params, n_sources,) = get_full_params_from_tile_params(
            tile_locs_sampled.reshape(
                n_samples * n_image_ptiles, -1, 2
            ),  # 2 = len((x,y))
            tile_source_params_sampled.reshape(
                n_samples * n_image_ptiles, -1, self.n_source_params
            ),
            tile_coords,
            slen,
            self.ptile_slen,
            self.edge_padding,
        )

        return locs, source_params, n_sources

    def _sample_tile_params(
        self,
        image,
        n_samples,
        return_map_n_sources,
        return_map_source_params,
        tile_n_sources,
        training,
    ):
        """
        NOTE: In the case of stars this will return log_fluxes!
        """

        assert (
            image.shape[0] == 1
        ), "Sampling only works for one image at a time for now..."

        # shape = (n_ptiles x n_bands x ptile_slen x ptile_slen)
        image_ptiles = self.get_image_ptiles(image, locs=None, source_params=None)[0]

        # pass through NN
        # shape = (n_ptiles x dim_out_all)
        h = self._get_var_params_all(image_ptiles)

        # get log probs for number of sources
        # shape = (n_ptiles x max_detections)
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(h)

        if not training:
            h = h.detach()
            log_probs_n_sources_per_tile = log_probs_n_sources_per_tile.detach()

        # sample number of stars
        # output shape = (n_samples x n_ptiles)
        if tile_n_sources is None:
            if return_map_n_sources:
                tile_n_stars_sampled = (
                    torch.argmax(log_probs_n_sources_per_tile.detach(), dim=1)
                    .repeat(n_samples)
                    .view(n_samples, -1)
                )

            else:
                tile_n_stars_sampled = _sample_class_weights(
                    torch.exp(log_probs_n_sources_per_tile.detach()), n_samples
                ).view(n_samples, -1)
        else:
            tile_n_stars_sampled = tile_n_sources.repeat(n_samples).view(n_samples, -1)

        is_on_array = get_is_on_from_tile_n_sources_2d(
            tile_n_stars_sampled, self.max_detections
        )
        # shape = (n_samples x n_ptiles x max_detections x 1 )
        is_on_array = is_on_array.unsqueeze(3).float()

        # get variational parameters: these are on image tiles
        # loc_mean.shape = (n_samples x n_ptiles x max_detections x len(x,y))
        (
            loc_mean,
            loc_logvar,
            source_param_mean,
            source_param_logvar,
        ) = self._get_var_params_for_n_sources(h, tile_n_stars_sampled)

        if return_map_source_params:
            loc_sd = torch.zeros(loc_logvar.shape, device=device)
            source_params_sd = torch.zeros(source_param_logvar.shape, device=device)
        else:
            loc_sd = torch.exp(0.5 * loc_logvar)
            source_params_sd = torch.exp(0.5 * source_param_logvar).clamp(max=0.5)

        # sample locations
        # shape = (n_samples x n_ptiles x max_detections x len(x,y))
        assert loc_mean.shape == loc_sd.shape, "Shapes need to match"
        tile_locs_sampled = torch.normal(loc_mean, loc_sd) * is_on_array

        # sample source params, these are log_fluxes or latent galaxy params (normal variables)
        assert source_param_mean.shape == source_params_sd.shape, "Shapes need to match"
        tile_source_params_sampled = torch.normal(source_param_mean, source_params_sd)

        return tile_locs_sampled, tile_source_params_sampled, is_on_array

    def sample_encoder(
        self,
        image,
        n_samples=1,
        return_map_n_sources=False,
        return_map_source_params=False,
        tile_n_sources=None,
        training=False,
    ):
        """
        In the case of stars, this function will return log_fluxes as source_params. Can then obtain
        fluxes with the following procedure:

        >> is_on_array = get_is_on_from_n_stars(n_stars, max_stars)
        >> fluxes = np.exp(log_fluxes) * is_on_array

        where `max_stars` corresponds to the maximum number of stars in a scene that was used when
        simulating the `image` passed in to this function.

        Args:
            image:
            n_samples:
            return_map_n_sources:
            return_map_source_params:
            tile_n_sources:
            training:

        Returns:

        """

        self.eval()
        if training:
            self.train()

        slen = image.shape[-1]
        (
            tile_locs_sampled,
            tile_source_params_sampled,
            is_on_array,
        ) = self._sample_tile_params(
            image,
            n_samples,
            return_map_n_sources,
            return_map_source_params,
            tile_n_sources,
            training,
        )
        tile_source_params_sampled *= is_on_array

        # get parameters on full image
        locs, source_params, n_sources = self._get_full_params_from_sampled_params(
            tile_locs_sampled, tile_source_params_sampled, slen
        )

        # returns either galaxy_params or log_fluxes.
        return locs, source_params, n_sources
