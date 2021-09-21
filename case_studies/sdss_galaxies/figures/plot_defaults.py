# flake8: noqa
import matplotlib as mpl


def set_rc_params(
    figsize=(10, 10),
    fontsize=18,
    label_size="medium",
    title_size="large",
    tick_label_size="small",
    legend_fontsize="medium",
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


def format_plot(ax, lims=None, ticks=None, xlabel=None, ylabel=None):

    # format plots.
    if lims:
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    if ticks:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
