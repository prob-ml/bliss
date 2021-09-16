# flake8: noqa
import matplotlib as mpl
from matplotlib import pyplot as plt

plt.style.use("seaborn-colorblind")


mpl.rcParams.update(
    {
        # figure
        "figure.figsize": (10, 10),
        # axes
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        # ticks
        "xtick.major.size": 7,
        "xtick.minor.size": 4,
        "xtick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "xtick.labelsize": 14,
        "ytick.major.size": 7,
        "ytick.minor.size": 6,
        "ytick.major.width": 0.8,
        "ytick.minor.width": 0.6,
        "ytick.labelsize": 14,
        # legend
        "legend.fontsize": 12,
    }
)

# font = {"family": "sans-serif", "sans-serif": ["Helvetica"]}
# plt.rc("font", **font)

plt.rc("text", usetex=False)
