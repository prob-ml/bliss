import json
import time

import numpy as np
import torch
import torch.optim as optim

from . import sleep_lib, sourcenet_lib, wake_lib, psf_transform_lib
from .data import simulated_datasets_lib
from .utils import const

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("torch version: ", torch.__version__)


def set_seed():
    np.random.seed(65765)
    _ = torch.manual_seed(3453453)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_params():
    with open("../data/default_star_parameters.json", "r") as fp:
        data_params = json.load(fp)
    print(data_params)
    return data_params


def load_psf():
    bands = [2, 3]
    psfield_file = (
        "./../celeste_net/sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit"
    )
    init_psf_params = psf_transform_lib.get_psf_params(psfield_file, bands=bands)
    power_law_psf = psf_transform_lib.PowerLawPSF(init_psf_params.cuda())
    psf_og = power_law_psf.forward().detach()

    return bands, psf_og


def load_background(bands, data_params):
    # sky intensity: for the r and i band

    init_background_params = torch.zeros(len(bands), 3).cuda()
    init_background_params[:, 0] = torch.Tensor([686.0, 1123.0])
    planar_background = wake_lib.PlanarBackground(
        image_slen=data_params["slen"],
        init_background_params=init_background_params.cuda(),
    )
    background = planar_background.forward().detach()
    return background


def get_dataset(
    n_images,
    psf_og,
    data_params,
    background,
    transpose_psf=False,
    add_noise=True,
    draw_poisson=True,
):
    star_dataset = simulated_datasets_lib.StarsDataset.load_dataset_from_params(
        n_images,
        data_params,
        psf_og,
        background,
        transpose_psf=transpose_psf,
        add_noise=add_noise,
        draw_poisson=draw_poisson,
    )

    return star_dataset


def get_optimizer():
    learning_rate = 1e-3
    weight_decay = 1e-5
    optimizer = optim.Adam(
        [{"params": star_encoder.parameters(), "lr": learning_rate}],
        weight_decay=weight_decay,
    )
    return optimizer


def train(star_encoder, dataset, optimizer):
    n_epochs = 201
    print_every = 20
    print("training")

    out_path = const.reports_path.joinpath("results_2020-04-13")
    out_path.mkdir(exist_ok=True, parents=True)
    out_filename = out_path.joinpath("starnet_ri.dat")

    sleep_lib.run_sleep(
        star_encoder,
        dataset,
        optimizer,
        n_epochs,
        out_filename=out_filename,
        print_every=print_every,
    )


def main():
    with torch.cuda.device(device):
        set_seed()
        data_params = load_data_params()
        bands, psf_og = load_psf()
        background = load_background(bands, data_params)

        # setup dataset.
        n_images = 200
        star_dataset = get_dataset(n_images, psf_og, data_params, background)
        star_dataset.cuda()

        star_encoder = sourcenet_lib.SourceEncoder(
            slen=data_params["slen"],
            patch_slen=8,
            step=2,
            edge_padding=3,
            n_bands=psf_og.shape[0],
            max_detections=2,
        )
        star_encoder.cuda()

        optimizer = get_optimizer()

        train(star_encoder, star_dataset, optimizer)


if __name__ == "__main__":
    main()
