# galsim_decoder changes for coadds
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
import galsim
import torch
import torch.nn.functional as F
from torch import Tensor
from bliss.models.decoder import get_mgrid

def align_single_exposures(img0, img1, slen, pixel_scale, dithers):
    img0 = galsim.Image(np.array(img0), wcs=galsim.PixelScale(pixel_scale)) if type(img0) is not galsim.image.Image else img0
    wcs0 = img0.wcs
    images = img1

    sgrid = (get_mgrid(slen) - (-1))/(1-(-1)) * (slen)
    grid_x = wcs0.xyTouv(np.array(sgrid.reshape(slen*slen,2)[:,0]), np.array(sgrid.reshape(slen*slen,2)[:,1]))[0]
    grid_y = wcs0.xyTouv(np.array(sgrid.reshape(slen*slen,2)[:,0]), np.array(sgrid.reshape(slen*slen,2)[:,1]))[1]

    grid = torch.empty(size=(0, 2))
    for i in dithers:
        wcs1 = galsim.OffsetWCS(scale = 0.393, origin = galsim.PositionD(i))
        x, y = wcs1.uvToxy(grid_x,grid_y)
        x_grid = (x/slen) * (1-(-1)) + (-1)
        y_grid = (y/slen) * (1-(-1)) + (-1)
        grid = torch.cat([grid, torch.stack((torch.tensor(x_grid),torch.tensor(y_grid)),-1)], dim=0)

    iplots = []
    input = torch.tensor(images[:]).reshape(len(dithers),1,slen,slen).float()
    grids = grid.reshape(len(dithers),1,slen*slen,2).float()
    iplots.append(F.grid_sample(input, grids, align_corners = False))
    
    tenplot = torch.tensor(iplots[:][0])
    crop_images = []
    for i in range(tenplot.shape[0]):
        im = np.array(tenplot[i].reshape(slen,slen))
        crop_im = im[1:slen-1, 1:slen-1]
        crop_images.append(crop_im)
    return crop_images