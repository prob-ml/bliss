import pickle
from pathlib import Path
import pandas as pd
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
    bliss_output_path = "/data/scratch/declan/redshift/dc2/plots/continuous_split_0_2025-04-18-14-39-44/cts_mode_metrics_0thbest_01.pkl"
    disc_run_name = "discrete_split_0_2025-04-22-09-50-08"
    bliss_discrete_output_path = output_dir / disc_run_name / "discrete_mode_metrics_0thbest.pkl"
    bliss_discrete_grid_output_path = output_dir / disc_run_name / "discrete_grid_metrics_0thbest.pkl"

    # Get bspline 
    bspline_run_name = "bspline_split_0_2025-04-22-23-17-28"
    bliss_bspline_output_path = output_dir / bspline_run_name / "bspline_mode_metrics_0thbest_01.pkl"

    # Get mdn
    mdn_run_name = "mdn_split_0_2025-04-22-23-17-15"
    bliss_mdn_output_path = output_dir / mdn_run_name / "mdn_mode_metrics_0thbest_01.pkl"

    # Get discrete metrics
    with open(bliss_discrete_output_path, "rb") as inputp:
        bliss_mode_out_dict = pickle.load(inputp)
    with open(bliss_discrete_grid_output_path, "rb") as inputp:
        bliss_discrete_out_dict = pickle.load(inputp)

    # Do continuous metrics
    with open(bliss_output_path, "rb") as inputp:
        bliss_out_dict = pickle.load(inputp)

    # Do bspline metrics
    with open(bliss_bspline_output_path, "rb") as inputp:
        bliss_bspline_out_dict = pickle.load(inputp)
    # Do mdn metrics
    with open(bliss_mdn_output_path, "rb") as inputp:
        bliss_mdn_out_dict = pickle.load(inputp)


    metrics = ["outlier_fraction_cata", "outlier_fraction", "l1", "mse"]
    metric_labels = [
        "Catastrophic Outlier Fraction",
        "Outlier Fraction",
        "Absolute Bias (L1)",
        "Mean Squared Error (L2)",
    ]
    sns.set_theme()

    # Get all binned metrics to compute
    metric_names = set(["_".join(x.split("_")[:-1]) for x in bliss_out_dict.keys() if "bin" in x])
    metric_names.remove('redshift_bias_bin_mag_redshifts/bias_bin')
    metric_names.remove('redshift_bias_bin_rs_redshifts/bias_bin')
    metric_names.remove('redshift_abs_bias_bin_mag_redshifts/bias_abs_bin')
    metric_names.remove('redshift_abs_bias_bin_rs_redshifts/bias_abs_bin')

    rail_results = pd.read_csv("/data/scratch/declan/redshift/dc2/checkpoints/rail/results.csv")
    metric_names.remove("redshifts/bias_abs_bin")
    metric_names.remove("redshifts/bias_bin")

    save_dir = output_dir / run_name / "mdnb_01_normal"
    save_dir.mkdir(parents=True, exist_ok=True)
    for this_metric in metric_names:
        if "rs" in this_metric:
            continue
        save_as = this_metric.split("/")[0]
        save_name = save_dir / f"{save_as}_{run_name}_log.pdf"

        # Get loss type mapping to rail
        if "l1" in this_metric:
            rail_metric = "L1"
        elif "mse" in this_metric:
            rail_metric = "L2"
        elif "cata" in this_metric:
            rail_metric = "catastrophic"
        elif "outlier_fraction_bin" in this_metric:
            rail_metric = "one_plus"
        else:
            raise ValueError(f"Unknown metric: {this_metric}")
        
        bin_labels = ["<23.5", "23.5-23.6", "23.6-23.7", "23.7-23.8", "23.8-23.9", "23.9-24.0", "24.0-24.1", "24.1-24.2", "24.2-24.3", "24.3-24.4", "24.4-24.5", "24.5-24.6", "24.6-24.7", "24.7-24.8", "24.8-24.9", ">24.9"]
        # if "rs" in this_metric.split("_"):
        #     bin_labels = ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1", ">1"]
        #     rail_nums = rail_results[(rail_results["loss_type"] == rail_metric) & (rail_results["binning_name"] == 'redshift') & (rail_results["split"] == 0)]
        # elif "mag" in this_metric.split("_"):
        #     bin_labels = ["<23.5", "23.5-23.6", "23.6-23.7", "23.7-23.8", "23.8-23.9", "23.9-24.0", "24.0-24.1", "24.1-24.2", "24.2-24.3", "24.3-24.4", "24.4-24.5", "24.5-24.6", "24.6-24.7", "24.7-24.8", "24.8-24.9", ">24.9"]

        #     rail_nums = rail_results[(rail_results["loss_type"] == rail_metric) & (rail_results["binning_name"] == 'mag') & (rail_results["split"] == 0)]
        # else:
        #     raise ValueError(f"Unknown metric: {this_metric}")

        num_values = len(bin_labels)
        start_name = "redshifts/" + this_metric.split('/')[-1]
        bliss_values = [bliss_out_dict[f"{this_metric}_{i}"] for i in range(num_values)]
        # bliss_discrete = [bliss_mode_out_dict[f"{this_metric}_{i}"] for i in range(num_values)]
        # bliss_discrete_grid = [
        #     bliss_discrete_out_dict[f"{this_metric}_{i}"] for i in range(num_values)
        # ]
        
        bliss_bspline = [bliss_bspline_out_dict[f"{start_name}_{i}"] for i in range(num_values)]
        bliss_mdn = [bliss_mdn_out_dict[f"{start_name}_{i}"] for i in range(num_values)]

        plt.figure(figsize=(6, 6))
        plt.plot(bin_labels, bliss_values, label="BLISS+Normal", marker="o", c="blue")
        # plt.plot(bin_labels, bliss_discrete, label="BLISS+Discrete Bin", marker="o", c="green")
        # plt.plot(
        #     bin_labels,
        #     bliss_discrete_grid,
        #     label="BLISS+Discrete Bin w/ Grid Search",
        #     marker="o",
        #     c="orange",
        # )
        # plt.plot(bin_labels, rail_nums["loss"].values, label="FlexZBoost", marker="o", c="red")
        plt.plot(bin_labels, bliss_bspline, label="BLISS+BSpline", marker="o", c="aqua")
        plt.plot(bin_labels, bliss_mdn, label="BLISS+MDN", marker="o", c="fuchsia")

        xlabel = "Redshift Bin" if "rs" in this_metric else "Magnitude Bin"
        ylabel = None
        for i, metric in enumerate(metrics):
            if metric in this_metric:
                ylabel = metric_labels[i]
                break
        plt.xlabel(xlabel)
        plt.xticks(rotation=45)
        plt.ylabel(ylabel)
        # plt.ylim([0, None])
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.grid(True)
        plt.yscale("log")
        plt.savefig(save_name)


if __name__ == "__main__":
    main()
