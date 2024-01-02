import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from torchmetrics import Metric


class TwoPointMetric(Metric):
    radii = torch.tensor([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])

    def __init__(self):
        super().__init__()
        self.add_state("obs_neighbors", default=torch.zeros(len(self.radii)))
        self.add_state("expected_neighbors", default=torch.zeros(len(self.radii)))
        self.add_state("n_est", default=torch.zeros(1))

    def update(self, _true_cat, est_cat, _matching):
        for i in range(est_cat.batch_size):
            ne_all = est_cat.n_sources[i].item()
            locs_all = est_cat.plocs[i, :ne_all]

            is_inner = ((locs_all > 10) & (locs_all < 98)).all(dim=1)
            locs_inner = locs_all[is_inner]

            ne_inner = locs_inner.shape[0]
            self.n_est += ne_inner

            kd_all = cKDTree(locs_all.cpu().numpy())
            kd_inner = cKDTree(locs_inner.cpu().numpy())

            n_obs = kd_inner.count_neighbors(kd_all, self.radii) - ne_inner
            self.obs_neighbors += torch.from_numpy(n_obs).to(self.device)

            other_per_pixel = (ne_all - 1) / 108**2  # adjust for image size (108 x 108 outer)
            other_per_disk = other_per_pixel * torch.pi * self.radii**2
            n_expected = other_per_disk * ne_inner
            self.expected_neighbors += n_expected.to(self.device)

    def compute(self):
        two_pt = (self.obs_neighbors / self.expected_neighbors) - 1
        return {f"{r}": two_pt[i] for i, r in enumerate(self.radii)}

    def plot(self):
        two_pt = (self.obs_neighbors / self.expected_neighbors) - 1
        two_pt = two_pt.cpu().numpy()

        fig, ax = plt.subplots()
        sns.lineplot(x=self.radii, y=two_pt, ax=ax, marker="s")
        ax.set_xscale("log")
        ax.set_xlabel("Radius")
        ax.set_ylabel("Two-point correlation")

        plt.tight_layout()
        plt.show()

        return fig, ax
