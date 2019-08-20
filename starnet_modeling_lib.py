import numpy as np
import timeit

import matplotlib.pyplot as plt

import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import normal

def get_bernoulli_entropy(p, lb = 1e-8):
    return -p * torch.log(p + lb) - (1 - p) * torch.log(1 - p + lb)

def get_log_pareto_prior(x, alpha, lb = 1e-8):
    return - (alpha + 1) * torch.log(x + lb)

def get_image_loglik(image, image_mean):
    assert list(image.shape) == list(image_mean.shape)
    assert torch.all(image_mean >= 0)
    return -(image - image_mean)**2 / (2 * image_mean) - 0.5 * torch.log(image_mean)

def get_bernoulli_loglik(x, p, lb = 1e-8):
    assert list(x.shape) == list(p.shape)
    return x * torch.log(p + lb) + (1 - x) * torch.log(1 - p + lb)

def get_normal_pdf(x, mean, scale):
    normal_distr = normal.Normal(loc = mean, scale = scale)
    return torch.exp(normal_distr.log_prob(x))

def get_logit_norm_entropy(sample, mean, log_var, lb = 1e-8):
    # sample should be drawn from a logit normal distribution

    # normal entropy
    norm_entropy = 0.5 * log_var

    # jacobian term
    log_jac_sample = torch.log(1 - sample + lb) + \
                        torch.log(sample + lb)

    logit_sample = torch.log(sample + lb) - \
                    torch.log(1 - sample + lb)

    scale = torch.exp(0.5 * log_var)

    return norm_entropy + log_jac_sample * \
                get_normal_pdf(logit_sample, mean, scale)

def get_log_norm_entropy(log_sample, mean, log_var, lb = 1e-8):
    # sample should be drawn from a log notmal distribution

    # normal entropy
    norm_entropy = 0.5 * log_var

    # jacobian term
    # print('log flux sample: ', log_sample)
    scale = torch.exp(0.5 * log_var)
    return norm_entropy + log_sample * \
            get_normal_pdf(log_sample, mean, scale)
