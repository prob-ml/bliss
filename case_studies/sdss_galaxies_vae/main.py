#!/usr/bin/env python3
import hydra

from bliss.train import train


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    main()
