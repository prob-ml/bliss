# flake8: noqa
import matplotlib as mpl

CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def set_rc_params(
    figsize=(10, 10),
    fontsize=18,
    title_size="large",
    label_size="medium",
    legend_fontsize="medium",
    tick_label_size="small",
    major_tick_size=7,
    minor_tick_size=4,
    major_tick_width=0.8,
    minor_tick_width=0.6,
):
    # named size options: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    mpl.rcParams.update(
        {
            # font.
            "font.family": "STIXGeneral",
            "font.sans-serif": "Helvetica",
            "text.usetex": True,
            "mathtext.fontset": "cm",
            "font.size": fontsize,
            # figure
            "figure.figsize": figsize,
            # axes
            "axes.labelsize": label_size,
            "axes.titlesize": title_size,
            # ticks
            "xtick.labelsize": tick_label_size,
            "ytick.labelsize": tick_label_size,
            "xtick.major.size": major_tick_size,
            "ytick.major.size": major_tick_size,
            "xtick.major.width": major_tick_width,
            "ytick.major.width": major_tick_width,
            "ytick.minor.size": minor_tick_size,
            "xtick.minor.size": minor_tick_size,
            "xtick.minor.width": minor_tick_width,
            "ytick.minor.width": minor_tick_width,
            # legend
            "legend.fontsize": legend_fontsize,
        }
    )


def format_plot(ax, xlims=None, ylims=None, xticks=None, yticks=None, xlabel="", ylabel=""):
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
