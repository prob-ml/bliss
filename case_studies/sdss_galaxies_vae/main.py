#!/usr/bin/env python3
import hydra

from bliss.train import train
from reconstruction import reconstruct


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    mode = cfg.mode
    if mode == "train":
        train(cfg)
    elif mode == "reconstruct":
        reconstruct(cfg)


if __name__ == "__main__":
    main()
