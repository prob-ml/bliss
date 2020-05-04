# TODO: Experiment to see what works best as a step size.
#       revisit for the case of wake phase need to be more careful.

import json

import numpy as np
import torch
import torch.optim as optim

from . import sleep_lib
from .models import sourcenet_lib
from .data import simulated_datasets_lib
from .utils import const

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def set_seed():
    np.random.seed(65765)
    _ = torch.manual_seed(3453453)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_params():
    parameters_path = const.data_path.joinpath("default_galaxy_parameters.json")
    with open(parameters_path, "r") as fp:
        data_params = json.load(fp)
    return data_params


def get_optimizer(galaxy_encoder):
    learning_rate = 1e-3
    weight_decay = 1e-5
    optimizer = optim.Adam(
        [{"params": galaxy_encoder.parameters(), "lr": learning_rate}],
        weight_decay=weight_decay,
    )
    return optimizer


def prepare_filepaths():
    out_dir = const.reports_path.joinpath("results_galaxy_2020-04-20")
    out_dir.mkdir(exist_ok=True, parents=True)

    state_dict_file = out_dir.joinpath("galaxy_i.dat")
    print(f"state dict file: {state_dict_file.as_posix()}")

    output_file = out_dir.joinpath(
        "output.txt"
    )  # save the output that is being printed.
    print(f"output file: {output_file.as_posix()}")

    return state_dict_file, output_file


def train(galaxy_encoder, dataset, optimizer, state_dict_file, output_file):
    n_epochs = 201
    print_every = 20
    batchsize = 32
    print("training")

    sleep_phase = sleep_lib.GalaxySleep(
        galaxy_encoder,
        dataset,
        n_epochs,
        galaxy_encoder.n_source_params,
        state_dict_file=state_dict_file,
        output_file=output_file,
        optimizer=optimizer,
        batchsize=batchsize,
        print_every=print_every,
    )

    print(f"running sleep phase for n_epochs = {n_epochs}, batchsize={batchsize}")
    sleep_phase.run_sleep()


def main():
    with torch.cuda.device(device):
        set_seed()
        data_params = load_data_params()

        state_dict_file, output_file = prepare_filepaths()

        # setup dataset.
        n_images = 320
        galaxy_dataset = simulated_datasets_lib.GalaxyDataset.load_dataset_from_params(
            n_images, data_params
        )

        galaxy_encoder = sourcenet_lib.GalaxyEncoder(
            slen=data_params["slen"],
            n_bands=1,
            patch_slen=20,
            step=5,
            edge_padding=5,
            max_detections=2,
            n_source_params=galaxy_dataset.simulator.latent_dim,
        ).cuda()

        optimizer = get_optimizer(galaxy_encoder)

        train(galaxy_encoder, galaxy_dataset, optimizer, state_dict_file, output_file)


if __name__ == "__main__":
    main()
