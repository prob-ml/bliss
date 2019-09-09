import torch
import torch.nn as nn

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)

class MyInstanceNorm1d(nn.Module):
    def __init__(self, d):
        super(MyInstanceNorm1d, self).__init__()

        self.instance_norm = nn.InstanceNorm1d(d, track_running_stats=False)

    def forward(self, tensor):
        assert len(tensor.shape) == 2
        return self.instance_norm(tensor.unsqueeze(1)).squeeze()


class StarEncoder(nn.Module):
    def __init__(self, slen, n_bands, max_detections):

        super(StarEncoder, self).__init__()

        # image parameters
        self.slen = slen
        self.n_bands = n_bands

        # max number of detections
        self.max_detections = max_detections

        # convolutional NN paramters
        enc_conv_c = 20
        enc_kern = 3
        enc_hidden = 256

        momentum = 0.1

        # convolutional NN
        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.n_bands, enc_conv_c, enc_kern,
                        stride=1, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern,
                        stride=1, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern,
                        stride=1, padding=1),
            nn.BatchNorm2d(enc_conv_c, momentum = momentum, track_running_stats=True),
            nn.ReLU(),
            Flatten()
        )

        # output dimension of convolutions
        conv_out_dim = \
            self.enc_conv(torch.zeros(1, n_bands, slen, slen)).size(1)

        # fully connected layers
        self.enc_fc = nn.Sequential(
            nn.Linear(conv_out_dim, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum = momentum, track_running_stats=True),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum = momentum, track_running_stats=True),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum = momentum, track_running_stats=True),
            nn.ReLU(),
        )

        # add final layer, whose size depends on the number of stars to output
        for i in range(0, max_detections + 1):
            # i = 0, 1, ..., max_detections
            len_out = i * 6 + 1
            width_hidden = len_out * 10

            self.add_module('enc_a_detect' + str(i),
                            nn.Sequential(nn.Linear(enc_hidden, width_hidden),
                                    nn.ReLU(),
                                    nn.Linear(width_hidden, width_hidden),
                                    nn.BatchNorm1d(width_hidden, momentum = momentum, track_running_stats=True),
                                    nn.ReLU()))

            self.add_module('enc_b_detect' + str(i),
                            nn.Sequential(nn.Linear(width_hidden + enc_hidden, width_hidden),
                                    nn.ReLU(),
                                    nn.Linear(width_hidden, width_hidden),
                                    nn.BatchNorm1d(width_hidden, momentum = momentum, track_running_stats=True),
                                    nn.ReLU()))

            final_module_name = 'enc_final_detect' + str(i)
            self.add_module(final_module_name, nn.Linear(2 * width_hidden + enc_hidden, len_out))

        # there are self.max_detections * (self.max_detections + 1)
        #    total possible detections, and each detection has
        #    seven parameters (thhee means, three variances, for two locs and one
        #    flux; and one probability)
        self.dim_out_all = \
            int(0.5 * self.max_detections * (self.max_detections + 1)  * 6 + \
                    1 + self.max_detections)
        self._get_hidden_indices()

        # self.enc_final = nn.Linear(enc_hidden, self.dim_out_all)
        self.log_softmax = nn.LogSoftmax(dim = 1)

    def _forward_to_pooled_hidden(self, image, background):
        h = self.enc_conv(torch.log(image - background + 1000.))
        return self.enc_fc(h)

    def _forward_conditional_nstars(self, h, n_stars):
        assert isinstance(n_stars, int)

        h_a = getattr(self, 'enc_a_detect' + str(n_stars))(h)
        h_b = getattr(self, 'enc_b_detect' + str(n_stars))(torch.cat((h_a, h), dim = 1))
        h_c = getattr(self, 'enc_final_detect' + str(n_stars))(torch.cat((h_a, h_b, h), dim = 1))

        return h_c

    def _forward_to_last_hidden(self, image, background):
        h = self._forward_to_pooled_hidden(image, background)

        h_out = torch.zeros(image.shape[0], 1).to(device)
        for i in range(0, self.max_detections + 1):
            h_i = self._forward_conditional_nstars(h, i)

            h_out = torch.cat((h_out, h_i), dim = 1)

        return h_out[:, 1:h_out.shape[1]]

    def forward(self, images, background, n_stars):
        # pass through neural network
        h = self._forward_to_last_hidden(images, background)

        # extract parameters
        logit_loc_mean, logit_loc_logvar, \
                log_flux_mean, log_flux_logvar, free_probs = \
                    self._get_params_from_last_hidden_layer(h, n_stars)

        log_probs = self.log_softmax(free_probs)

        return logit_loc_mean, logit_loc_logvar, \
                log_flux_mean, log_flux_logvar, log_probs


    def _get_params_from_last_hidden_layer(self, h, n_stars):

        assert h.shape[1] == self.dim_out_all
        assert h.shape[0] == len(n_stars)

        batchsize = h.size(0)
        _h = torch.cat((h, torch.zeros(batchsize, 1).to(device)), dim = 1)

        logit_loc_mean = torch.gather(_h, 1, self.locs_mean_indx_mat[n_stars])
        logit_loc_logvar = torch.gather(_h, 1, self.locs_var_indx_mat[n_stars])

        log_flux_mean = torch.gather(_h, 1, self.fluxes_mean_indx_mat[n_stars])
        log_flux_logvar = torch.gather(_h, 1, self.fluxes_var_indx_mat[n_stars])

        free_probs = h[:, self.prob_indx]

        return logit_loc_mean.reshape(batchsize, self.max_detections, 2), \
                logit_loc_logvar.reshape(batchsize, self.max_detections, 2), \
                log_flux_mean.reshape(batchsize, self.max_detections), \
                log_flux_logvar.reshape(batchsize, self.max_detections), \
                free_probs


    def _get_hidden_indices(self):

        self.locs_mean_indx_mat = \
            torch.full((self.max_detections + 1, 2 * self.max_detections),
                        self.dim_out_all).type(torch.LongTensor).to(device)

        self.locs_var_indx_mat = \
            torch.full((self.max_detections + 1, 2 * self.max_detections),
                        self.dim_out_all).type(torch.LongTensor).to(device)

        self.fluxes_mean_indx_mat = \
            torch.full((self.max_detections + 1, self.max_detections),
                        self.dim_out_all).type(torch.LongTensor).to(device)
        self.fluxes_var_indx_mat = \
            torch.full((self.max_detections + 1, self.max_detections),
                        self.dim_out_all).type(torch.LongTensor).to(device)

        self.prob_indx = torch.zeros(self.max_detections + 1).type(torch.LongTensor).to(device)

        for n_detections in range(1, self.max_detections + 1):
            indx0 = int(0.5 * n_detections * (n_detections - 1) * 6) + \
                            (n_detections  - 1) + 1

            indx1 = (2 * n_detections) + indx0
            indx2 = (2 * n_detections) * 2 + indx0

            # indices for locations
            self.locs_mean_indx_mat[n_detections, 0:(2 * n_detections)] = torch.arange(indx0, indx1)
            self.locs_var_indx_mat[n_detections, 0:(2 * n_detections)] = torch.arange(indx1, indx2)

            indx3 = indx2 + n_detections
            indx4 = indx3 + n_detections

            # indices for fluxes
            self.fluxes_mean_indx_mat[n_detections, 0:n_detections] = torch.arange(indx2, indx3)
            self.fluxes_var_indx_mat[n_detections, 0:n_detections] = torch.arange(indx3, indx4)

            self.prob_indx[n_detections] = indx4
