import torch


def convert_mag_to_nmgy(mag):
    return 10 ** ((22.5 - mag) / 2.5)


def convert_nmgy_to_mag(nmgy):
    return 22.5 - 2.5 * torch.log10(nmgy)
