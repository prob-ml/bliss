import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)

class StarCNN(nn.Module):
    def __init__(self, slen, n_bands):

        super(StarCNN, self).__init__()

        self.slen = slen
        self.n_bands = n_bands

        enc_conv_c = 20
        enc_kern = 5

        self.enc_conv = nn.Sequential(
            nn.Conv2d(n_bands, enc_conv_c, enc_kern,
                        stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern,
                        stride=1, padding=0),
            nn.ReLU(),
            Flatten()
        )

    def forward(self, image):

        h = self.enc_conv(image)

        return(h)

class StarEncoder(nn.Module):
    def __init__(self, h_length):

        super(StarEncoder, self).__init__()

        self.h_length = h_length

        enc_hidden = 256

        self.enc_fc = nn.Sequential(
            nn.Linear(h_length, enc_hidden),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, track_running_stats=False),
            nn.ReLU(),

            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, track_running_stats=False),
            nn.ReLU(),

            nn.Linear(enc_hidden, 6)
        )

    def forward(self, h):
        h = self.enc_fc(h)

        logit_locs_mean = h[:, 0:2]
        logit_locs_logvar = h[:, 2:4]

        log_flux_mean = h[:, 4]
        log_flux_logvar = h[:, 5]

        return logit_locs_mean, logit_locs_logvar, \
                    log_flux_mean, log_flux_logvar

class StarRNN(nn.Module):
    def __init__(self, slen, n_bands):

        super(StarRNN, self).__init__()

        self.slen = slen
        self.n_bands = n_bands

        self.flatten = Flatten()

        self.star_cnn = StarCNN(slen, n_bands)

        h_out = self.star_cnn(torch.zeros(1, n_bands, slen, slen))
        self.hidden_length = h_out.shape[1]

        self.h_length = 2 * self.hidden_length + slen**2

        self.star_enc = StarEncoder(self.h_length)

    def forward_once(self, image_i, h_i):
        h_new = self.star_cnn(image_i)

        return self.star_enc(torch.cat((self.flatten(image_i), h_i, h_new), 1))
