import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def update_device(new_device):
    global device
    device = new_device
    if use_cuda:
        torch.cuda.set_device(device)
