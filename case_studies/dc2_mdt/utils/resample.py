from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from case_studies.dc2_mdt.utils.gaussian_diffusion import GaussianDiffusion
from case_studies.dc2_mdt.utils.reverse_markov_learning import RMLDiffusion


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def sample_weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - batch_sample_weights: a tensor of weights to scale the final losses.
                 - batch_loss_weights: a tensor of weights to scale the MSE losses.
        """
        w = self.sample_weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        batch_sample_weights = th.from_numpy(weights_np).float().to(device)
        return indices, batch_sample_weights, None


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self._sample_weights = np.ones([diffusion.num_timesteps])

    def sample_weights(self):
        return self._sample_weights
    

class SpeedSampler(ScheduleSampler):
    def __init__(self, 
                 diffusion: GaussianDiffusion,
                 lam,
                 k,
                 tau):
        self.diffusion = diffusion
        assert isinstance(self.diffusion, GaussianDiffusion)

        grad = np.gradient(self.diffusion.sqrt_one_minus_alphas_cumprod)
        self.meaningful_steps = np.argmax(grad < 1e-4) + 1

        self.lam = lam
        self.k = k
        self.tau = tau

        higher = self.k
        lower = 1
        p = [higher] * self.tau + \
            [lower] * (self.diffusion.num_timesteps - self.tau)
        self.p = F.normalize(th.tensor(p, dtype=th.float32), p=1, dim=0)

        w = np.gradient(1 - self.diffusion.alphas_cumprod)
        one_minus_lam = 1 - self.lam
        w = one_minus_lam + (1 - 2 * one_minus_lam) * (w - w.min()) / (w.max() - w.min())
        w = w[: self.tau].tolist() + \
            [1] * (self.diffusion.num_timesteps - self.tau)
        self.loss_weights = th.tensor(w)

    def sample_weights(self):
        raise NotImplementedError()
    
    def sample(self, batch_size, device):
        t = th.multinomial(self.p, batch_size // 2 + 1, replacement=True)
        dual_t = th.where(t < self.meaningful_steps, 
                          self.meaningful_steps - t, 
                          t - self.meaningful_steps)
        t = th.cat([t, dual_t], dim=0)[:batch_size].long().to(device=device)
        return t, th.ones_like(t).float(), self.loss_weights.to(device=device)[t]
    

class SigmoidSampler(ScheduleSampler):
    def __init__(self, 
                 diffusion: RMLDiffusion,
                 b):
        assert isinstance(diffusion, RMLDiffusion)
        alpha_2 = diffusion.alpha ** 2 + 1e-3
        sigma_2 = diffusion.sigma ** 2 + 1e-3
        self.num_timesteps = diffusion.num_timesteps
        self.loss_weights = 1 / (1 + th.exp(b - th.log(alpha_2 / sigma_2)))

    def sample_weights(self):
        raise NotImplementedError()
    
    def sample(self, batch_size, device):
        t_np = np.random.choice(self.num_timesteps, size=(batch_size,))
        t = th.from_numpy(t_np).long().to(device)
        return t, th.ones_like(t).float(), self.loss_weights.to(device=device)[t]


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def sample_weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
