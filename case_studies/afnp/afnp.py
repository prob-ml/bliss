#%%
import sys
import argparse
import os 
import torch
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('/home/dereklh/projects/bliss2')
import bliss
import bliss.datasets.sdss as sdss
from bin.utils import add_path_args, setup_paths
#%%
sdss_obj = sdss.SloanDigitalSkySurvey(Path("/home/dereklh/projects/bliss2/data/sdss/"), run=3900, camcol=6, field=269, bands=range(5))

bright_stars = sdss_obj[1]['bright_stars']


# %%
plt.imshow(bright_stars[4])
# %%
plt.imshow(z)
# %%
