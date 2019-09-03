import torch
import torch.nn as nn

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)

class StarEncoder(nn.Module):
    def __init__(self, slen, n_bands, max_detections):

        super(StarEncoder, self).__init__()

        # image parameters
        self.slen = slen
        self.n_bands = n_bands

        # max number of detections
        self.max_detections = max_detections

        # convolutional NN parameters
        enc_conv_c = 32
        enc_kern = 5
        enc_hidden = 256

        # convolutional NN
        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.n_bands, enc_conv_c, enc_kern,
                        stride=1, padding=3),
            nn.ReLU(),

            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern,
                        stride=1, padding=3),
            nn.ReLU(),

            Flatten()
        )

        # output dimension of convolutions
        conv_out_dim = \
            self.enc_conv(torch.zeros(1, n_bands, slen, slen)).size(1)

        # fully connected layers
        self.enc_fc = nn.Sequential(
            nn.Linear(conv_out_dim, enc_hidden),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, track_running_stats=True),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, track_running_stats=True),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.ReLU(),
        )

        # add final layer, whose size depends on the number of stars to output
        for i in range(1, max_detections + 1):
            module_name = 'enc_final_detect' + str(i)

            len_out = i * 6
            module = nn.Sequential(nn.Linear(enc_hidden, len_out * 10),
                                    nn.ReLU(),
                                    nn.Linear(len_out * 10, len_out),
                                    nn.ReLU())
            self.add_module(module_name, module)


        # there are self.max_detections * (self.max_detections + 1)
        #    total possible detections, and each detection has
        #    six parameters (trhee means, three variances, for two locs and one
        #    flux)
        self.dim_out_all = \
            int(0.5 * self.max_detections * (self.max_detections + 1)  * 6)

        # self.enc_final = nn.Linear(enc_hidden, self.dim_out_all)

    def forward_to_last_hidden(self, images, backgrounds):
        h = self.enc_conv(images - backgrounds)
        h = self.enc_fc(h)

        h_out = torch.zeros(images.shape[0], 1)
        for i in range(1, self.max_detections + 1):
            h_out = torch.cat((h_out,
                        getattr(self, 'enc_final_detect' + str(i))(h)),
                        axis = 1)

        return h_out[:, 1:h_out.shape[1]]

    def forward(self, images, backgrounds, n_stars):
        h = self.forward_to_last_hidden(images, backgrounds)

        batchsize = images.shape[0]

        # get parameters: last layer depends on the number of stars
        logit_loc_mean, logit_loc_logvar, \
                log_flux_mean, log_flux_logvar = (0, 0, 0, 0)

        for i in range(1, self.max_detections + 1):
            logit_loc_mean_i, logit_loc_logvar_i, \
                log_flux_mean_i, log_flux_logvar_i = \
                    self._get_params_from_last_hidden_layer(h, i)

            is_on = (n_stars == i).float()
            logit_loc_mean = logit_loc_mean + \
                is_on.view(batchsize, 1, 1) * logit_loc_mean_i
            logit_loc_logvar = logit_loc_logvar + \
                is_on.view(batchsize, 1, 1) * logit_loc_logvar_i

            log_flux_mean = log_flux_mean + \
                is_on.view(batchsize, 1) * log_flux_mean_i
            log_flux_logvar = log_flux_logvar + \
                is_on.view(batchsize, 1) * log_flux_logvar_i

        return logit_loc_mean, logit_loc_logvar, \
                log_flux_mean, log_flux_logvar


    def _get_params_from_last_hidden_layer(self, h, n_detections):

        assert h.shape[1] == self.dim_out_all
        assert n_detections <= self.max_detections
        assert n_detections > 0

        batchsize = h.size(0)

        indx0 = int(0.5 * n_detections * (n_detections - 1) * 6)

        indx1 = (2 * n_detections) + indx0
        indx2 = (2 * n_detections) * 2 + indx0

        # get locations
        logit_loc_mean = h[:, indx0:indx1].view(-1, n_detections, 2)
        logit_loc_logvar = h[:, indx1:indx2].view(-1, n_detections, 2)

        indx3 = indx2 + n_detections
        indx4 = indx3 + n_detections

        # get fluxes
        log_flux_mean = h[:, indx2:indx3]
        log_flux_logvar = h[:, indx3:indx4]

        # pad params
        if not n_detections == self.max_detections:
            locs_pad_size = (batchsize, self.max_detections - n_detections, 2)
            pad_locs = torch.full(locs_pad_size, 0.).to(device)
            logit_loc_mean = torch.cat((logit_loc_mean, pad_locs), dim = 1)
            logit_loc_logvar = torch.cat((logit_loc_logvar, pad_locs), dim = 1)

            fluxes_pad_size = (batchsize, self.max_detections - n_detections)
            pad_flux = torch.full(fluxes_pad_size, 0.).to(device)
            log_flux_mean = torch.cat((log_flux_mean, pad_flux), dim = 1)
            log_flux_logvar = torch.cat((log_flux_logvar, pad_flux), dim = 1)

        return logit_loc_mean, logit_loc_logvar, \
                log_flux_mean, log_flux_logvar

class StarCounter(nn.Module):
    def __init__(self, slen, n_bands, max_detections):

        super(StarCounter, self).__init__()

        conv_len = 16 * int(np.ceil(slen / 8)) ** 2

        self.max_detections = max_detections

        enc_hidden = 128

        self.detector = nn.Sequential(
            nn.Conv2d(n_bands, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),

            Flatten(),

            nn.Linear(conv_len, enc_hidden),
            nn.BatchNorm1d(enc_hidden, track_running_stats=False),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, track_running_stats=False),
            nn.ReLU(),

            nn.Linear(enc_hidden, self.max_detections + 1),
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, image):
        return self.detector(image)
