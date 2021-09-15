import matplotlib as mpl
from matplotlib import pyplot as plt

plt.style.use("seaborn-colorblind")


mpl.rcParams.update(
    {
        # figure
        "figure.figsize": (10, 10),
        # axes
        "axes.labelsize": 24,
        "axes.titlesize": 28,
        # ticks
        "xtick.major.size": 10,
        "xtick.minor.size": 5,
        "xtick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "xtick.labelsize": 22,
        "ytick.major.size": 10,
        "ytick.minor.size": 5,
        "ytick.major.width": 0.8,
        "ytick.minor.width": 0.6,
        "ytick.labelsize": 22,
        # legend
        "legend.fontsize": 22,
    }
)
plt.rc("font", family="sans-serif", sans_serif=["Helvetica"])
plt.rc("text", usetex=True)
