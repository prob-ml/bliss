from hydra import initialize, compose
from hydra.utils import instantiate
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, linregress
import pytorch_lightning as pl
from pytorch_lightning.utilities import move_data_to_device
from bliss.global_env import GlobalEnv
import os
import sys
import yaml
import time
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Navigate to the BLISS root directory
bliss_root = '/gpfs/accounts/regier_root/regier0/taodingr/bliss'
os.chdir(bliss_root)

# Add BLISS root to Python path
if bliss_root not in sys.path:
    sys.path.insert(0, bliss_root)

print("New working directory:", os.getcwd())

def draw_credible_interval(ckpt, setting):
    ckpt = ckpt
    setting = setting

    # test data path in congif_descwl, make sure to change that each time !!!
    with initialize(config_path="../", version_base=None):
        cfg = compose("config_descwl", {
            "train.pretrained_weights=" + ckpt,
            })

    seed = pl.seed_everything(cfg.train.seed)
    GlobalEnv.seed_in_this_program = seed

    data_source = instantiate(cfg.train.data_source)
    data_source.setup("test")
    test_dl = data_source.test_dataloader()

    # Load in encoder weights
    encoder = instantiate(cfg.encoder).to(device)
    encoder_state_dict = torch.load(cfg.train.pretrained_weights, map_location=device)["state_dict"]
    encoder.load_state_dict(encoder_state_dict)
    encoder = encoder.eval()

    confidence_levels = torch.linspace(0.05, 0.95, steps = 19)
    ci_quantiles = torch.distributions.Normal(0, 1).icdf(1 - (1 - confidence_levels)/2).to(device)

    shear1_true = torch.zeros(len(test_dl), device='cpu')
    
    shear2_true = torch.zeros(len(test_dl), device='cpu')

    shear1_ci_lower = torch.zeros(len(test_dl), len(ci_quantiles), device='cpu')
    shear1_ci_upper = torch.zeros(len(test_dl), len(ci_quantiles), device='cpu')
    shear2_ci_lower = torch.zeros(len(test_dl), len(ci_quantiles), device='cpu')
    shear2_ci_upper = torch.zeros(len(test_dl), len(ci_quantiles), device='cpu')

    i = -1
    for b in test_dl:
        i += 1
        batch = move_data_to_device(b, device)
        
        with torch.no_grad():
            shear1_true[i] = batch['tile_catalog']['shear_1'].squeeze().flatten()
            shear2_true[i] = batch['tile_catalog']['shear_2'].squeeze().flatten()
            
            input_lst = [inorm.get_input_tensor(batch) for inorm in encoder.image_normalizers]
            inputs = torch.cat(input_lst, dim=2)

            x_cat_marginal = encoder.net(inputs).squeeze()
            
            shear1_ci_lower[i] = x_cat_marginal[0] - ci_quantiles * x_cat_marginal[1].exp().sqrt()
            shear1_ci_upper[i] = x_cat_marginal[0] + ci_quantiles * x_cat_marginal[1].exp().sqrt()
            shear2_ci_lower[i] = x_cat_marginal[2] - ci_quantiles * x_cat_marginal[3].exp().sqrt()
            shear2_ci_upper[i] = x_cat_marginal[2] + ci_quantiles * x_cat_marginal[3].exp().sqrt()

    shear1_coverage_probs = ((shear1_ci_lower <= shear1_true.unsqueeze(-1)) * (shear1_true.unsqueeze(-1) <= shear1_ci_upper)).float().mean(0)
    shear2_coverage_probs = ((shear2_ci_lower <= shear2_true.unsqueeze(-1)) * (shear2_true.unsqueeze(-1) <= shear2_ci_upper)).float().mean(0)

    for i, ci in enumerate(confidence_levels):
        print(f'Confidence level: {ci:.2f}, Shear 1: {shear1_coverage_probs[i]:.4f}, Shear 2: {shear2_coverage_probs[i]:.4f}')

    # plot coverage plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    fontsize = 20
    ticklabelsize = 16
    color = 'darkorchid'
    s = 80

    _ = ax[0].axline((0,0), slope = 1, linestyle = 'dotted', color = 'black', linewidth=2, zorder=0)
    _ = ax[0].scatter(confidence_levels, shear1_coverage_probs, color=color, s=s, zorder=1)
    _ = ax[0].set_title('$\gamma_1$', fontsize=1.5*fontsize)
    _ = ax[0].set_xlabel('Nominal coverage probability', fontsize = fontsize)
    _ = ax[0].set_ylabel('Empirical coverage probability', fontsize = fontsize)
    _ = ax[0].tick_params(axis='both', which='major', labelsize=ticklabelsize)
    _ = ax[0].set_xlim(0, 1)
    _ = ax[0].set_ylim(0, 1)

    _ = ax[1].axline((0,0), slope = 1, linestyle = 'dotted', color = 'black', linewidth=2, zorder=0)
    _ = ax[1].scatter(confidence_levels, shear2_coverage_probs, color=color, s=s, zorder=1)
    _ = ax[1].set_title('$\gamma_2$', fontsize=1.5*fontsize)
    _ = ax[1].set_xlabel('Nominal coverage probability', fontsize = fontsize)
    _ = ax[1].set_ylabel('Empirical coverage probability', fontsize = fontsize)
    _ = ax[1].tick_params(axis='both', which='major', labelsize=ticklabelsize)
    _ = ax[1].set_xlim(0, 1)
    _ = ax[1].set_ylim(0, 1)

    for a in ax.flat:
        _ = a.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()

    fig.savefig(f"/scratch/regier_root/regier0/taodingr/bliss/case_studies/weak_lensing/descwl/notebooks/figures/{setting}_coverageprobs.png", dpi = 300, transparent = True, bbox_inches = 'tight', pad_inches = 0)

    # plot credible interval plot
    fig, ax = plt.subplots(1, 2, figsize=(13,6))
    fontsize = 20
    ticklabelsize = 16
    color = np.array(['darkgoldenrod','darkorchid'])
    alpha = 0.5

    axmin = min(shear1_ci_lower.min(), shear2_ci_lower.min()).cpu() - 0.01
    axmax = max(shear1_ci_upper.max(), shear2_ci_upper.max()).cpu() + 0.01

    np.random.seed(0)
    indexes = np.arange(len(test_dl))

    interval_idx = 17 # 90% credible interval

    _ = ax[0].axline((0,0), slope = 1, linestyle = 'dotted', color = 'black', linewidth=2)
    shear1_coverage = (
        ((shear1_ci_lower <= shear1_true.unsqueeze(-1)
            ) * (shear1_true.unsqueeze(-1) <= shear1_ci_upper))[...,interval_idx]
        )

    covered_legend = False
    uncovered_legend = False

    for i in range(len(test_dl)):
        if (shear1_coverage[indexes[i]]) and (not covered_legend):
            covered_legend = True
            _ = ax[0].vlines(x = shear1_true[indexes[i]],
                            ymin = shear1_ci_lower[...,interval_idx][indexes[i]],
                            ymax = shear1_ci_upper[...,interval_idx][indexes[i]],
                            alpha = alpha, color = color[shear1_coverage[indexes[i]]],
                            label = f"covers ({round(100 * shear1_coverage_probs[interval_idx].item(), 1)}%)")
        elif (not shear1_coverage[indexes[i]]) and (not uncovered_legend):
            uncovered_legend = True
            _ = ax[0].vlines(x = shear1_true[indexes[i]],
                            ymin = shear1_ci_lower[...,interval_idx][indexes[i]],
                            ymax = shear1_ci_upper[...,interval_idx][indexes[i]],
                            alpha = alpha, color = color[shear1_coverage[indexes[i]]],
                            label = f"does not cover ({round(100 * (1 - shear1_coverage_probs[interval_idx].item()), 1)}%)")
        else:
            _ = ax[0].vlines(x = shear1_true[indexes[i]],
                            ymin = shear1_ci_lower[...,interval_idx][indexes[i]],
                            ymax = shear1_ci_upper[...,interval_idx][indexes[i]],
                            alpha = alpha, color = color[shear1_coverage[indexes[i]]])
    _ = ax[0].set_xlabel('$\gamma_1$', fontsize=fontsize)
    _ = ax[0].set_ylabel(r'$\widehat{\gamma}_1$', fontsize=fontsize)
    _ = ax[0].tick_params(axis='both', which='major', labelsize=ticklabelsize)
    _ = ax[0].legend(loc = 'upper left', prop = {'size': ticklabelsize})
    _ = ax[0].set_xlim(axmin, axmax)
    _ = ax[0].set_ylim(axmin, axmax)



    _ = ax[1].axline((0,0), slope = 1, linestyle = 'dotted', color = 'black', linewidth=2)
    shear2_coverage = (
        ((shear2_ci_lower <= shear2_true.unsqueeze(-1)
            ) * (shear2_true.unsqueeze(-1) <= shear2_ci_upper))[...,interval_idx]
        )

    covered_legend = False
    uncovered_legend = False

    for i in range(len(indexes)):
        if (shear2_coverage[indexes[i]]) and (not covered_legend):
            covered_legend = True
            _ = ax[1].vlines(x = shear2_true[indexes[i]],
                            ymin = shear2_ci_lower[...,interval_idx][indexes[i]],
                            ymax = shear2_ci_upper[...,interval_idx][indexes[i]],
                            alpha = alpha, color = color[shear2_coverage[indexes[i]]],
                            label = f"covers ({round(100 * shear2_coverage_probs[interval_idx].item(), 1)}%)")
        elif (not shear2_coverage[indexes[i]]) and (not uncovered_legend):
            uncovered_legend = True
            _ = ax[1].vlines(x = shear2_true[indexes[i]],
                            ymin = shear2_ci_lower[...,interval_idx][indexes[i]],
                            ymax = shear2_ci_upper[...,interval_idx][indexes[i]],
                            alpha = alpha, color = color[shear2_coverage[indexes[i]]],
                            label = f"does not cover ({round(100 * (1 - shear2_coverage_probs[interval_idx].item()), 1)}%)")
        else:
            _ = ax[1].vlines(x = shear2_true[indexes[i]],
                            ymin = shear2_ci_lower[...,interval_idx][indexes[i]],
                            ymax = shear2_ci_upper[...,interval_idx][indexes[i]],
                            alpha = alpha, color = color[shear2_coverage[indexes[i]]])
    _ = ax[1].set_xlabel('$\gamma_2$', fontsize=fontsize)
    _ = ax[1].set_ylabel(r'$\widehat{\gamma}_2$', fontsize=fontsize)
    _ = ax[1].tick_params(axis='both', which='major', labelsize=ticklabelsize)
    _ = ax[1].legend(loc = 'upper left', prop = {'size': ticklabelsize})
    _ = ax[1].set_xlim(axmin, axmax)
    _ = ax[1].set_ylim(axmin, axmax)

    for a in ax.flat:
        _ = a.spines[['top', 'right']].set_visible(False)

    _ = fig.tight_layout()

    fig.savefig(f"/scratch/regier_root/regier0/taodingr/bliss/case_studies/weak_lensing/descwl/notebooks/figures/{setting}_credibleintervals.png", dpi = 300, transparent = True, bbox_inches = 'tight', pad_inches = 0)

def draw_scatter(ckpt, setting):
    ckpt=ckpt
    setting=setting

    with initialize(config_path="../", version_base=None):
        cfg = compose("config_descwl", {
            "train.pretrained_weights=" + ckpt,
            })

    seed = pl.seed_everything(cfg.train.seed)
    GlobalEnv.seed_in_this_program = seed

    data_source = instantiate(cfg.train.data_source)
    data_source.setup("test")
    test_dl = data_source.test_dataloader()

    encoder = instantiate(cfg.encoder).to(device)
    encoder_state_dict = torch.load(cfg.train.pretrained_weights, map_location=device)["state_dict"]
    encoder.load_state_dict(encoder_state_dict)
    encoder = encoder.eval()

    shear1_true = torch.zeros(len(test_dl), device=device)
    shear1_pred = torch.zeros(len(test_dl), device=device)
    shear2_true = torch.zeros(len(test_dl), device=device)
    shear2_pred = torch.zeros(len(test_dl), device=device)
    test_loss = torch.zeros(len(test_dl), device = device)

    i = -1
    for b in test_dl:
        i += 1
        batch = move_data_to_device(b, device)
        
        shear1_true[i] = batch['tile_catalog']['shear_1'].squeeze()
        shear2_true[i] = batch['tile_catalog']['shear_2'].squeeze()
        
        with torch.no_grad():
            mode_cat = encoder.sample(batch, use_mode=True)
            test_loss[i] = encoder._compute_loss(batch, None)
        
        shear1_pred[i] = mode_cat['shear_1'].squeeze()
        shear2_pred[i] = mode_cat['shear_2'].squeeze()

    print(f'shear 1 test MSE (BLISS) = {((shear1_true - shear1_pred) ** 2).mean()}')
    print(f'shear 2 test MSE (BLISS) = {((shear2_true - shear2_pred) ** 2).mean()}')

    print(f'shear 1 pearson correlation (BLISS) = {np.corrcoef(shear1_true.flatten().cpu(), shear1_pred.flatten().cpu())[1,0]}')
    print(f'shear 2 pearson correlation (BLISS) = {np.corrcoef(shear2_true.flatten().cpu(), shear2_pred.flatten().cpu())[1,0]}')

    print(f'shear 1 spearman correlation (BLISS) = {spearmanr(shear1_true.flatten().cpu(), shear1_pred.flatten().cpu())[0]}')
    print(f'shear 2 spearman correlation (BLISS) = {spearmanr(shear2_true.flatten().cpu(), shear2_pred.flatten().cpu())[0]}')

    print(f'shear 1 kendall correlation (BLISS) = {kendalltau(shear1_true.flatten().cpu(), shear1_pred.flatten().cpu())[0]}')
    print(f'shear 2 kendall correlation (BLISS) = {kendalltau(shear2_true.flatten().cpu(), shear2_pred.flatten().cpu())[0]}')

    print(f'test loss = {test_loss.mean()}')

    alpha = 0.5
    s = 50
    fontsize = 20
    ticklabelsize = 16
    bliss_color = 'darkorchid'
    baseline_color = 'sienna'
    axmin = min(shear1_true.min(), shear2_true.min(), shear1_pred.min(), shear2_pred.min()).cpu() - 0.005
    axmax = max(shear1_true.max(), shear2_true.max(), shear1_pred.max(), shear2_pred.max()).cpu() + 0.005

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    lr1 = linregress(shear1_true.flatten().cpu().numpy(), shear1_pred.flatten().cpu().numpy())
    lr2 = linregress(shear2_true.flatten().cpu().numpy(), shear2_pred.flatten().cpu().numpy())

    # Shear 1
    _ = ax[0].scatter(shear1_true.flatten().cpu().numpy(),
                    shear1_pred.flatten().cpu().numpy(),
                    color=bliss_color, alpha=alpha, s=s, zorder=1)
    x_line = np.linspace(axmin, axmax, 100)
    y_line1 = lr1.slope * x_line + lr1.intercept
    ax[0].plot(x_line, y_line1, color='black', linewidth=2, alpha=0.4, zorder=2)

    _ = ax[0].set_xlabel(r'True $\gamma_1$', fontsize=fontsize)
    _ = ax[0].set_ylabel(r'$\widehat{\gamma}_1$ (posterior mean)', fontsize=fontsize)
    _ = ax[0].tick_params(axis='both', which='major', labelsize=ticklabelsize)
    _ = ax[0].set_xlim((axmin, axmax))
    _ = ax[0].set_ylim((axmin, axmax))

    # Shear 2
    _ = ax[1].scatter(shear2_true.flatten().cpu().numpy(),
                    shear2_pred.flatten().cpu().numpy(),
                    color=bliss_color, alpha=alpha, s=s, zorder=1)
    y_line2 = lr2.slope * x_line + lr2.intercept
    ax[1].plot(x_line, y_line2, color='black', linewidth=2, alpha=0.4, zorder=2)

    _ = ax[1].set_xlabel(r'True $\gamma_2$', fontsize=fontsize)
    _ = ax[1].set_ylabel(r'$\widehat{\gamma}_2$ (posterior mean)', fontsize=fontsize)
    _ = ax[1].tick_params(axis='both', which='major', labelsize=ticklabelsize)
    _ = ax[1].set_xlim((axmin, axmax))
    _ = ax[1].set_ylim((axmin, axmax))

    for a in ax.flat:
        _ = a.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()

    fig.savefig(f"/scratch/regier_root/regier0/taodingr/bliss/case_studies/weak_lensing/descwl/notebooks/figures/{setting}_scatterplots.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)

    print(f"Shear 1:\nc ± 3SE = {lr1.intercept:.6f} ± {3 * lr1.intercept_stderr:.6f}, m ± 3SE = {lr1.slope - 1:.6f} ± {3*lr1.stderr}\n")
    print(f"Shear 2:\nc ± 3SE = {lr2.intercept:.6f} ± {3 * lr2.intercept_stderr:.6f}, m ± 3SE = {lr2.slope - 1:.6f} ± {3*lr2.stderr}")

def main():
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"=== Credible Interval generation started ===")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 50)
    with open('/scratch/regier_root/regier0/taodingr/bliss/case_studies/weak_lensing/descwl/notebooks/credibleinterval_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    draw_credible_interval(ckpt=config['ckpt'], setting=config['setting'])
    draw_scatter(ckpt=config['ckpt'], setting=config['setting'])

    end_time = time.time()
    end_datetime = datetime.now()
    print(f"\n=== SIMULATION COMPLETED ===")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"=" * 50)

if __name__ == "__main__":
    main()
