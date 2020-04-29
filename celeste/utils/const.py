import torch
import numpy as np
from pathlib import Path
from os.path import dirname

from torch.distributions import categorical


src_path = Path(dirname(dirname(__file__)))
root_path = Path(dirname(dirname(dirname(__file__))))

data_path = root_path.joinpath("data")
reports_path = root_path.joinpath("reports")
models_path = root_path.joinpath("models")


def get_is_on_from_n_sources(n_sources, max_sources):
    """
    Return a boolean array of shape=(batchsize, max_sources) whose (k,l)th entry indicates
    whether there are more than l sources on the kth batch.
    :param n_sources:
    :param max_sources:
    :return:
    """
    assert len(n_sources.shape) == 1

    batchsize = len(n_sources)
    is_on_array = (
        torch.cuda.LongTensor(batchsize, max_sources).zero_()
        if torch.cuda.is_available()
        else torch.zeros((batchsize, max_sources), dtype=torch.long)
    )
    for i in range(max_sources):
        is_on_array[:, i] = n_sources > i

    return is_on_array


def get_is_on_from_patch_n_sources_2d(
    patch_n_sources, max_sources, device=torch.device("cuda")
):
    """

    :param patch_n_sources: A tensor of shape (n_samples x n_patches), indicating the number of sources
                            at sample i, batch j. (n_samples = batchsize)
    :type patch_n_sources: class: `torch.Tensor`
    :param max_sources:
    :param device:
    :type max_sources: int
    :return:
    """
    #
    assert not torch.any(torch.isnan(patch_n_sources))
    assert torch.all(patch_n_sources >= 0)
    assert torch.all(patch_n_sources <= max_sources)

    n_samples = patch_n_sources.shape[0]
    batchsize = patch_n_sources.shape[1]

    is_on_array = torch.zeros(
        (n_samples, batchsize, max_sources), dtype=torch.long, device=device
    )
    for i in range(max_sources):
        is_on_array[:, :, i] = patch_n_sources > i

    return is_on_array


def get_one_hot_encoding_from_int(z, n_classes):
    z = z.long()

    assert len(torch.unique(z)) <= n_classes

    z_one_hot = torch.zeros(len(z), n_classes).cuda()
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot


#############################
# Sampling functions
############################


def draw_pareto(f_min, alpha, shape):
    uniform_samples = torch.cuda.FloatTensor(*shape).uniform_()
    return f_min / (1.0 - uniform_samples) ** (1 / alpha)


def draw_pareto_maxed(f_min, f_max, alpha, shape):
    # draw pareto conditioned on being less than f_max

    pareto_samples = draw_pareto(f_min, alpha, shape)

    while torch.any(pareto_samples > f_max):
        indx = pareto_samples > f_max
        pareto_samples[indx] = draw_pareto(f_min, alpha, [torch.sum(indx).item()])

    return pareto_samples


def sample_class_weights(class_weights, n_samples=1):
    """
    Draw a sample from Categorical variable with
    probabilities class_weights.
    """

    assert not torch.any(torch.isnan(class_weights))
    cat_rv = categorical.Categorical(probs=class_weights)
    return cat_rv.sample((n_samples,)).detach().squeeze()


def sample_normal(mean, logvar):
    return mean + torch.exp(0.5 * logvar) * torch.randn(mean.shape).cuda()


#############################
# Log probabilities
############################


def _logit(x, tol=1e-8):
    return torch.log(x + tol) - torch.log(1 - x + tol)


def eval_logitnormal_logprob(x, mu, logvar):
    logit_x = _logit(x)
    return eval_normal_logprob(logit_x, mu, logvar)


def eval_normal_logprob(x, mu, logvar):
    return (
        -0.5 * logvar
        - 0.5 * (x - mu) ** 2 / (torch.exp(logvar) + 1e-5)
        - 0.5 * np.log(2 * np.pi)
    )


def eval_lognormal_logprob(x, mu, log_var, tol=1e-8):
    log_x = torch.log(x + tol)
    return eval_normal_logprob(log_x, mu, log_var)
