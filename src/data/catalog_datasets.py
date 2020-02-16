import numpy as np
from astropy.table import Table
from torch.utils.data import Dataset


class CatsimData(Dataset):

    def __init__(self):
        """
        This class reads the relevant parameters OneDegSq.fits file and returns samples from this
        ~800k row matrix.
        """
        super(CatsimData, self).__init__()

        # pa_disk = pa_bulge (by assumption)
        self.param_names = ['redshift',
                            'fluxnorm_bulge', 'fluxnorm_disk', 'fluxnorm_agn',
                            'a_b', 'a_d', 'b_b', 'b_d', 'pa_disk',
                            'u_ab', 'g_ab', 'r_ab', 'i_ab', 'z_ab', 'y_ab']
        self.num_params = len(self.param_names)

        self.table = Table.read("/home/imendoza/deblend/galaxy-net/params/OneDegSq.fits")
        np.random.shuffle(self.table)  # shuffle just in case order of galaxies matters in original table.
        self.params = self.table[self.param_names]  # array of tuples of len = 18.

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        return np.array([self.params[idx][i] for i in range(len(self.param_names))], dtype=np.float32)
