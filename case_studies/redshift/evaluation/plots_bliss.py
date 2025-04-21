import pickle
from pathlib import Path

import hydra
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from omegaconf import DictConfig
from hydra import compose, initialize


@hydra.main(config_path=".", config_name="continuous_eval")
def main(cfg: DictConfig):
    with initialize(config_path="."):
        cfg = compose(config_name="continuous_eval")
    output_dir = Path(cfg.paths.plot_dir)
    run_name = cfg.paths.ckpt_dir.split("/")[-2]
    # Load metric results
    bliss_output_path = output_dir / run_name / "cts_mode_metrics_0thbest_new_bins.pkl"
    disc_run_name = "discrete_split_0_2025-04-18-15-30-55"
    bliss_discrete_output_path = output_dir / disc_run_name / "discrete_mode_metrics_0thbest.pkl"
    bliss_discrete_grid_output_path = output_dir / disc_run_name / "discrete_grid_metrics_0thbest.pkl"

    # Get discrete metrics
    with open(bliss_discrete_output_path, "rb") as inputp:
        bliss_mode_out_dict = pickle.load(inputp)
    with open(bliss_discrete_grid_output_path, "rb") as inputp:
        bliss_discrete_out_dict = pickle.load(inputp)

    # Do continuous metrics
    with open(bliss_output_path, "rb") as inputp:
        bliss_out_dict = pickle.load(inputp)

    metrics = ["outlier_fraction_cata", "outlier_fraction", "bias", "bias_abs", "mse"]
    metric_labels = [
        "Catastrophic Outlier Fraction",
        "Outlier Fraction",
        "Bias",
        "Absolute Bias",
        "MSE",
    ]
    sns.set_theme()

    # Get all binned metrics to compute
    metric_names = set(["_".join(x.split("_")[:-1]) for x in bliss_out_dict.keys() if "bin" in x])


    for this_metric in metric_names:
        save_as = this_metric.split("/")[0]
        save_name = output_dir / run_name/ f"{save_as}_{run_name}.pdf"
        
        if "rs" in this_metric.split("_"):
            bin_labels = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", ">3"]
        elif "mag" in this_metric.split("_"):
            bin_labels = ["<23.9", "23.9-24.1", "24.1-24.5", "24.5-24.9", "24.9-25.6", ">25.6"]
        else:
            raise ValueError(f"Unknown metric: {this_metric}")

        num_values = len(bin_labels)
        bliss_values = [bliss_out_dict[f"{this_metric}_{i}"] for i in range(num_values)]
        bliss_discrete = [bliss_mode_out_dict[f"{this_metric}_{i}"] for i in range(num_values)]
        bliss_discrete_grid = [
            bliss_discrete_out_dict[f"{this_metric}_{i}"] for i in range(num_values)
        ]

        plt.figure(figsize=(6, 6))
        plt.plot(bin_labels, bliss_values, label="BLISS+Normal", marker="o", c="blue")
        plt.plot(bin_labels, bliss_discrete, label="BLISS+Discrete Bin", marker="o", c="green")
        plt.plot(
            bin_labels,
            bliss_discrete_grid,
            label="BLISS+Discrete Bin w/ Grid Search",
            marker="o",
            c="orange",
        )
        xlabel = "Redshift Bin" if "rs" in this_metric else "Magnitude Bin"
        ylabel = None
        for i, metric in enumerate(metrics):
            if metric in this_metric:
                ylabel = metric_labels[i]
                break
        plt.xlabel(xlabel)
        plt.xticks(rotation=45)
        plt.ylabel(ylabel)
        plt.ylim([0, None])
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(save_name)


if __name__ == "__main__":
    main()
