import torch
import torch.nn as nn
from torch.distributions import categorical
from .. import device


def get_is_on_from_n_sources(n_sources, max_sources):
    """Return a boolean array of shape=(batch_size, max_sources) whose (k,l)th entry indicates
    whether there are more than l sources on the kth batch.
    """
    assert not torch.any(torch.isnan(n_sources))
    assert torch.all(n_sources >= 0)
    assert torch.all(n_sources <= max_sources)

    is_on_array = torch.zeros(
        *n_sources.shape, max_sources, device=device, dtype=torch.float
    )

    for i in range(max_sources):
        is_on_array[..., i] = n_sources > i

    return is_on_array


def _argfront(is_on_array, dim):
    # return indices that sort pushing all zeroes of tensor to the back.
    # dim is dimension along which do the ordering.
    assert len(is_on_array.shape) == 2
    indx_sort = (is_on_array != 0).long().argsort(dim=dim, descending=True)
    return indx_sort


def _sample_class_weights(class_weights, n_samples=1):
    """
    Draw a sample from Categorical variable with
    probabilities class_weights.
    """

    assert not torch.any(torch.isnan(class_weights))
    cat_rv = categorical.Categorical(probs=class_weights)
    return cat_rv.sample((n_samples,)).squeeze()


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


def _clip_params(max_detections, tile_n_sources, *tiled_params):
    _tile_n_sources = tile_n_sources.clamp(max=max_detections)

    _tiled_params = []
    for tiled_param in tiled_params:
        _tiled_param = tiled_param[:, 0:max_detections, ...]
        _tiled_params.append(_tiled_param)
    return (_tile_n_sources, *_tiled_params)


def _tile_locs(tile_coords, slen, edge_padding, ptile_slen, locs):
    assert torch.all(locs <= 1.0)
    assert torch.all(locs >= 0.0)

    batch_size = locs.size(0)
    max_sources = locs.size(1)
    single_image_n_ptiles = tile_coords.size(0)
    total_ptiles = single_image_n_ptiles * batch_size  # across all batches.

    # _tile_coords shape = (1 x single_image_n_ptiles x 1 x 2)
    _tile_coords = tile_coords.unsqueeze(0).unsqueeze(2).float()
    left_tile_edges = _tile_coords + edge_padding - 0.5
    right_tile_edges = _tile_coords - 0.5 + ptile_slen - edge_padding
    locs = locs * (slen - 1)

    # indicator for each ptile, whether there is a loc there or not (loc order maintained)
    # .unsqueeze(1) in order to get repetition across `single_image_n_ptiles` dimension.
    # importantly, both have dim 2 at the end.
    tile_is_on_array = locs.unsqueeze(1) > left_tile_edges
    tile_is_on_array &= locs.unsqueeze(1) < right_tile_edges
    tile_is_on_array &= locs.unsqueeze(1) != 0
    tile_is_on_array = tile_is_on_array[:, :, :, 0] * tile_is_on_array[:, :, :, 1]
    tile_is_on_array = tile_is_on_array.float().to(device)
    tile_is_on_array = tile_is_on_array.view(total_ptiles, max_sources)

    # total number of sources in each tile.
    # .long() because use for indexing later.
    tile_n_sources = tile_is_on_array.sum(dim=1).long()

    # for each tile returned re-normalized locs in that tile, maintaining relative ordering of
    # locs including leading/trailing zeroes) in the case that there are multiple objects
    # in that tile.
    # need to .unsqueeze(0) because switching from batches to just tiles.
    _tile_is_on_array = tile_is_on_array.view(batch_size, -1, max_sources, 1)
    _locs = locs.view(batch_size, 1, max_sources, 2)
    tile_locs = _tile_is_on_array * _locs
    tile_locs = tile_locs.view(total_ptiles, max_sources, 2)
    _tile_coords = tile_coords.view(single_image_n_ptiles, 1, 2).repeat(
        batch_size, 1, 1
    )
    tile_locs -= _tile_coords + edge_padding - 0.5  # recenter
    tile_locs /= ptile_slen - 2 * edge_padding  # re-normalize
    tile_locs = torch.relu(tile_locs)  # some are negative now; set these to 0

    indx_sort = _argfront(tile_is_on_array, dim=1)
    tile_locs = torch.gather(tile_locs, 1, indx_sort.unsqueeze(2).repeat(1, 1, 2))

    return tile_n_sources, tile_locs, tile_is_on_array, indx_sort


def _get_tile_params(tile_is_on_array, indx_sort, params):
    # NOTE: indx_sort is an array that can order tile_is_on_array in some way with torch.gather

    total_ptiles = tile_is_on_array.size(0)
    max_sources = tile_is_on_array.size(1)

    tiled_params = []
    for param in params:
        assert max_sources == param.size(1)
        # this will work even for galaxy_bool
        batch_size = param.size(0)
        _param = param.view(batch_size, 1, max_sources, -1)
        param_dim = _param.size(-1)
        _tile_is_on_array = tile_is_on_array.view(batch_size, -1, max_sources, 1)

        tiled_param = _tile_is_on_array * _param
        tiled_param = tiled_param.view(total_ptiles, max_sources, -1)

        # now reorder.
        _indx_sort = indx_sort.unsqueeze(2).repeat(1, 1, param_dim)
        tiled_param = torch.gather(tiled_param, 1, _indx_sort)

        tiled_params.append(tiled_param)

    return tiled_params


def _get_params_in_tiles(
    tile_coords, max_detections, slen, edge_padding, ptile_slen, locs, *params
):
    (
        tile_n_sources,
        tile_locs,  # sorted.
        _tile_is_on_array,  # unsorted.
        indx_sort,
    ) = _tile_locs(tile_coords, slen, edge_padding, ptile_slen, locs)

    # now sort the rest of the parameters
    tile_params = _get_tile_params(_tile_is_on_array, indx_sort, params)

    # now that we are done we can sort this tensor too.
    tile_is_on_array = torch.gather(_tile_is_on_array, 1, indx_sort)

    # clip if necessary.
    (tile_n_sources, *tile_params, tile_is_on_array) = _clip_params(
        max_detections, tile_n_sources, tile_locs, *tile_params, tile_is_on_array
    )

    # tile_is_on_array shape = total_n_ptiles x max_detections
    assert len(tile_is_on_array.shape) == 2
    return (tile_n_sources, *tile_params, tile_is_on_array)


def _get_full_params_from_sampled_params(
    tile_coords,
    slen,
    ptile_slen,
    edge_padding,
    tile_is_on_array_sampled,
    tile_locs_sampled,
    *tile_params_sampled
):
    # NOTE: off sources should have tile_locs == 0.
    # NOTE: assume that each param in each tile is already pushed to the front.

    # tile_locs_sampled shape = (n_samples x n_ptiles x max_detections x 2)
    assert len(tile_locs_sampled.shape) == 4
    single_image_n_ptiles = tile_coords.shape[0]
    n_samples = tile_locs_sampled.shape[0]
    n_ptiles = tile_locs_sampled.shape[1]
    max_detections = tile_locs_sampled.shape[2]
    total_ptiles = n_samples * n_ptiles
    assert single_image_n_ptiles == n_ptiles, "Only single image is supported."

    n_sources = tile_is_on_array_sampled.sum(dim=(1, 2))  # per sample.
    max_sources = n_sources.max().int().item()

    # recenter and renormalize locations.
    tile_is_on_array = tile_is_on_array_sampled.view(total_ptiles, -1)
    tile_locs = tile_locs_sampled.view(total_ptiles, -1, 2)
    scale = ptile_slen - 2 * edge_padding
    bias = tile_coords.repeat(n_samples, 1).unsqueeze(1).float() + edge_padding - 0.5
    _locs = (tile_locs * scale + bias) / (slen - 1) * tile_is_on_array.unsqueeze(2)

    # sort locs and clip
    locs = _locs.view(n_samples, -1, 2)
    _indx_sort = _argfront(locs[..., 0], dim=1)
    indx_sort = _indx_sort.unsqueeze(2)
    locs = torch.gather(locs, 1, indx_sort.repeat(1, 1, 2))
    locs = locs[:, 0:max_sources, ...]

    # now do the same for the rest of the parameters (without scaling or biasing ofc)
    # for same reason no need to multiply times is_on_array
    params = []
    for tile_param_sampled in tile_params_sampled:
        # make sure works for galaxy bool too.
        assert len(tile_param_sampled.shape) == 4
        _param = tile_param_sampled.reshape(n_samples, n_ptiles, max_detections, -1)
        param_dim = _param.size(-1)
        param = _param.view(n_samples, -1, param_dim)
        param = torch.gather(param, 1, indx_sort.repeat(1, 1, param_dim))
        param = param[:, 0:max_sources, ...]

        params.append(param)

    return (n_sources, locs, *params)


def _extract_ptiles_2d(img, tile_shape, step, batch_first=False):
    """
    Take in an image (tensor) and the shape of the padded tile
    we want to separate it into and
    return the padded tiles also as a tensor.

    Taken from: https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78
    """

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

    :param images: A tensor of size (batch_size x n_bands x slen x slen)
    :param ptile_slen: The side length of each padded tile.
    :param step:
    :return: image_ptiles, output tensor of shape:
             (batch_size * ptiles per image) x n_bands x ptile_slen x ptile_slen
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
        image_ptiles = torch.cat((image_ptiles, image_ptiles_b.to(device)), dim=1)

    return image_ptiles


class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        slen,
        ptile_slen,
        step,
        edge_padding,
        n_bands,
        max_detections,
        n_galaxy_params,
        enc_conv_c=20,
        enc_kern=3,
        enc_hidden=256,
        momentum=0.5,
    ):
        """
        This class implements the source encoder, which is supposed to take in a synthetic image of
        size slen * slen
        and returns a NN latent variable representation of this image.

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
        super(ImageEncoder, self).__init__()

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

        self.momentum = momentum

        # convolutional NN
        conv_out_dim = self.enc_conv_c * ptile_slen ** 2
        self.enc_conv = nn.Sequential(
            nn.Conv2d(
                self.n_bands, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.enc_conv_c, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.BatchNorm2d(self.enc_conv_c, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(
                self.enc_conv_c, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.enc_conv_c, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.BatchNorm2d(self.enc_conv_c, momentum=self.momentum),
            nn.ReLU(),
            Flatten(),
            nn.Linear(conv_out_dim, self.enc_hidden),
            nn.BatchNorm1d(self.enc_hidden, momentum=self.momentum),
            nn.ReLU(),
            nn.Linear(self.enc_hidden, self.enc_hidden),
            nn.BatchNorm1d(self.enc_hidden, momentum=self.momentum),
            nn.ReLU(),
            nn.Linear(self.enc_hidden, self.enc_hidden),
            nn.BatchNorm1d(self.enc_hidden, momentum=self.momentum),
            nn.ReLU(),
        )

        # There are self.max_detections * (self.max_detections + 1)
        #  total possible detections, and each detection has
        #  4 + 2*n parameters (2 means and 2 variances for each loc + mean and variance for
        #  n source_param's (flux per band or galaxy params.) + 1 for the Bernoulli variable
        #  of whether the source is a star or galaxy.
        self.n_star_params = n_bands
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
            ("galaxy_params_mean", self.n_galaxy_params),
            ("galaxy_params_var", self.n_galaxy_params),
            ("log_fluxes_mean", self.n_star_params),
            ("log_fluxes_var", self.n_star_params),
            ("prob_galaxy", 1, lambda x: torch.sigmoid(x).clamp(1e-4, 1 - 1e-4)),
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

        # Concatenate all output parameters for all possible n_sources
        return self.enc_final(h)

    def forward(self, image_ptiles, n_sources):
        # will unsqueeze and squeeze n_sources later.
        assert len(n_sources.shape) == 1
        n_sources = n_sources.unsqueeze(0)  # will be used to index.

        # h.shape = (n_ptiles x self.dim_out_all)
        h = self._get_var_params_all(image_ptiles)

        # get probability of n_sources
        # shape = (n_ptiles x (max_detections+1))
        n_source_log_probs = self._get_logprob_n_from_var_params(h)

        # loc_mean has shape = (1 x n_ptiles x max_detections x len(x,y))
        (
            loc_mean,
            loc_logvar,
            galaxy_param_mean,
            galaxy_param_logvar,
            log_flux_mean,
            log_flux_logvar,
            prob_galaxy,
        ) = self._get_var_params_for_n_sources(
            h, n_sources=n_sources.clamp(max=self.max_detections)
        )

        # in the case of stars these are log_flux_mean, and log_flux_logvar.
        # squeeze if possible to account for non-sampling case.
        return {
            "n_source_log_probs": n_source_log_probs.squeeze(0),
            "loc_mean": loc_mean.squeeze(0),
            "loc_logvar": loc_logvar.squeeze(0),
            "galaxy_param_mean": galaxy_param_mean.squeeze(0),
            "galaxy_param_logvar": galaxy_param_logvar.squeeze(0),
            "log_flux_mean": log_flux_mean.squeeze(0),
            "log_flux_logvar": log_flux_logvar.squeeze(0),
            "prob_galaxy": prob_galaxy.squeeze(0),
        }

    def _get_tile_coords(self, slen):
        tile_coords = self.tile_coords

        # handle cases where images passed in are not of original size.
        if not (slen == self.slen):
            tile_coords = _get_tile_coords(slen, slen, self.ptile_slen, self.step)
        return tile_coords

    def get_params_in_tiles(self, slen, locs, *params):
        max_sources = locs.size(1)
        assert self.max_detections <= max_sources, "Wasteful, lower max_detections."

        tile_coords = self._get_tile_coords(slen)

        return _get_params_in_tiles(
            tile_coords,
            self.max_detections,
            slen,
            self.edge_padding,
            self.ptile_slen,
            locs,
            *params
        )

    def get_images_in_tiles(self, images):
        assert len(images.shape) == 4  # should be batch_size x n_bands x slen x slen
        assert images.size(1) == self.n_bands

        image_ptiles = _tile_images(images, self.ptile_slen, self.step)
        return image_ptiles

    def _get_full_params_from_sampled_params(
        self, slen, tile_is_on_array_sampled, tile_locs_sampled, *tile_params_sampled
    ):
        tile_coords = self._get_tile_coords(slen)
        return _get_full_params_from_sampled_params(
            tile_coords,
            slen,
            self.ptile_slen,
            self.edge_padding,
            tile_is_on_array_sampled,
            tile_locs_sampled,
            *tile_params_sampled
        )

    def _sample_tile_params(
        self, image, n_samples, return_map_n_sources, return_map_source_params,
    ):

        assert image.size(0) == 1, "Sampling only works for a single image."

        # shape = (n_ptiles x n_bands x ptile_slen x ptile_slen)
        image_ptiles = self.get_images_in_tiles(image)

        # shape = (n_ptiles x dim_out_all)
        h = self._get_var_params_all(image_ptiles)

        # shape = (n_ptiles x max_detections)
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(h)

        # sample number of sources.
        # output shape = (n_samples x n_ptiles)
        if return_map_n_sources:
            tile_n_sources_sampled = torch.argmax(log_probs_n_sources_per_tile, dim=1)
            tile_n_sources_sampled = tile_n_sources_sampled.repeat(n_samples)

        else:
            probs_n_sources_per_tile = torch.exp(log_probs_n_sources_per_tile)
            tile_n_sources_sampled = _sample_class_weights(
                probs_n_sources_per_tile, n_samples
            )
        tile_n_sources_sampled = tile_n_sources_sampled.view(n_samples, -1)

        # shape = (n_samples x n_ptiles x max_detections)
        tile_is_on_array = get_is_on_from_n_sources(
            tile_n_sources_sampled, self.max_detections
        )
        tile_is_on_array = tile_is_on_array.unsqueeze(3).float()

        # get variational parameters: these are on image tiles
        # shape (all) = (n_samples x n_ptiles x max_detections x param_dim)
        (
            loc_mean,
            loc_logvar,
            galaxy_param_mean,
            galaxy_param_logvar,
            log_flux_mean,
            log_flux_logvar,
            prob_galaxy,
        ) = self._get_var_params_for_n_sources(h, tile_n_sources_sampled)

        if return_map_source_params:
            tile_galaxy_bool_sampled = (prob_galaxy > 0.5).float()
            loc_sd = torch.zeros_like(loc_logvar)
            galaxy_param_sd = torch.zeros_like(galaxy_param_logvar)
            log_flux_sd = torch.zeros_like(log_flux_logvar)
        else:
            tile_galaxy_bool_sampled = torch.bernoulli(prob_galaxy).float()
            loc_sd = torch.exp(0.5 * loc_logvar)
            galaxy_param_sd = torch.exp(0.5 * galaxy_param_logvar)
            log_flux_sd = torch.exp(0.5 * log_flux_logvar)
        tile_galaxy_bool_sampled *= tile_is_on_array

        # shape = (n_samples x n_ptiles x max_detections x param_dim)
        assert loc_mean.shape == loc_sd.shape, "Shapes need to match"
        assert galaxy_param_mean.shape == galaxy_param_sd.shape, "Shapes need to match"
        assert log_flux_mean.shape == log_flux_sd.shape, "Shapes need to match"

        # TODO: For really bad initialization sometimes I get locs > 1, so clamp. Is this ok?
        tile_locs_sampled = torch.normal(loc_mean, loc_sd).clamp(0, 1)
        tile_locs_sampled *= tile_is_on_array

        tile_galaxy_params_sampled = torch.normal(galaxy_param_mean, galaxy_param_sd)
        tile_galaxy_params_sampled *= tile_is_on_array * tile_galaxy_bool_sampled

        tile_log_fluxes_sampled = torch.normal(log_flux_mean, log_flux_sd)
        tile_log_fluxes_sampled *= tile_is_on_array * (1 - tile_galaxy_bool_sampled)

        return (
            tile_locs_sampled,
            tile_galaxy_params_sampled,
            tile_log_fluxes_sampled,
            tile_galaxy_bool_sampled.squeeze(-1),
            tile_is_on_array.squeeze(-1),
        )

    def sample_encoder(
        self,
        image,
        n_samples=1,
        return_map_n_sources=False,
        return_map_source_params=False,
    ):
        # NOTE: make sure to use inside a `with torch.no_grad()` and with .eval() if applicable.

        # TODO: Check that this is always true?
        # returned sampled params already pushed to the front in each tile.
        slen = image.shape[-1]
        (
            tile_locs_sampled,
            tile_galaxy_params_sampled,
            tile_log_fluxes_sampled,
            tile_galaxy_bool_sampled,
            tile_is_on_array_sampled,
        ) = self._sample_tile_params(
            image, n_samples, return_map_n_sources, return_map_source_params,
        )

        # get parameters on full image
        (
            n_sources,
            locs,
            galaxy_params,
            log_fluxes,
            galaxy_bool,
        ) = self._get_full_params_from_sampled_params(
            slen,
            tile_is_on_array_sampled,
            tile_locs_sampled,
            tile_galaxy_params_sampled,
            tile_log_fluxes_sampled,
            tile_galaxy_bool_sampled.unsqueeze(-1),
        )

        return n_sources, locs, galaxy_params, log_fluxes, galaxy_bool
