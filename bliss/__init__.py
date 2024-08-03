# flake8: noqa
import omegaconf


def make_range(start, stop, step, *args):
    orig_range = list(range(start, stop, step))
    for arg in args:
        try:
            orig_range.remove(arg)
        except ValueError:
            continue
    return omegaconf.ListConfig(orig_range)


# resolve ranges in config files (we want this to execute at any entry point)
omegaconf.OmegaConf.register_new_resolver("range", make_range, replace=True)
