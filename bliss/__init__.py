# flake8: noqa
import omegaconf


def make_range(start, stop, step):
    return omegaconf.ListConfig(list(range(start, stop, step)))


# resolve ranges in config files (we want this to execute at any entry point)
omegaconf.OmegaConf.register_new_resolver("range", make_range, replace=True)
