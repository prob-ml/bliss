import pickle
from pathlib import Path

import hydra
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from omegaconf import DictConfig


@hydra.main(config_path=".", config_name="discrete_eval")
def main(cfg: DictConfig):
    output_dir = Path(cfg.paths.plot_dir)
    epoch = 5
    # Load metric results
    bliss_output_path = output_dir / "cts_mode_metrics_{}.pkl".format(epoch)
    bliss_discrete_output_path = output_dir / "discrete_mode_metrics_{}.pkl".format(epoch)
    bliss_discrete_grid_output_path = output_dir / "discrete_grid_metrics_{}.pkl".format(epoch)

    with open(bliss_discrete_output_path, "rb") as inputp:
        bliss_mode_out_dict = pickle.load(inputp)
    with open(bliss_discrete_grid_output_path, "rb") as inputp:
        bliss_discrete_out_dict = pickle.load(inputp)
    with open(bliss_output_path, "rb") as inputp:
        bliss_out_dict = pickle.load(inputp)

    metrics = ["outlier_fraction_cata", "outlier_fraction", "nmad", "bias_abs", "mse"]
    metric_labels = [
        "Catastrophic Outlier Fraction",
        "Outlier Fraction",
        "NMAD",
        "Absolute Bias",
        "MSE",
    ]
    sns.set_theme()
    for i, metric in enumerate(metrics):
        save_name = output_dir / f"{metric}_{epoch}.pdf"

        mag_ranges = ["<23.9", "23.9-24.1", "24.1-24.5", "24.5-24.9", "24.9-25.6", ">25.6"]
        bliss_values = [bliss_out_dict[f"redshifts/{metric}_bin_{i}"] for i in range(6)]
        bliss_discrete = [bliss_mode_out_dict[f"redshifts/{metric}_bin_{i}"] for i in range(6)]
        bliss_discrete_grid = [
            bliss_discrete_out_dict[f"redshifts/{metric}_bin_{i}"] for i in range(6)
        ]

        plt.figure(figsize=(6, 6))
        plt.plot(mag_ranges, bliss_values, label="BLISS+Normal", marker="o", c="blue")
        plt.plot(mag_ranges, bliss_discrete, label="BLISS+Discrete Bin", marker="o", c="green")
        plt.plot(
            mag_ranges,
            bliss_discrete_grid,
            label="BLISS+Discrete Bin w/ Grid Search",
            marker="o",
            c="orange",
        )
        plt.xlabel("Magnitude")
        plt.xticks(rotation=45)
        plt.ylabel(metric_labels[i])
        plt.ylim([0, None])
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(save_name)


if __name__ == "__main__":
    main()
