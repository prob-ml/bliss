import torch
import numpy as np

import matplotlib.pyplot as plt

def plot_image(axarr, image, x0 = 0, x1 = 0, slen0 = 100, slen1 = 100): 
    
    subimage = image[x0:(x0+slen0), x1:(x1+slen1)]
    vmin = subimage.min()
    vmax = subimage.max()
    
    im = axarr.matshow(image.cpu(), cmap = plt.cm.gray, vmin = vmin, vmax = vmax)
    axarr.set_ylim(x0 + slen0, x0)
    axarr.set_xlim(x1, x1 + slen1)
    
    return im

def plot_locations(locs, slen, ax, marker = 'o', color = 'b'): 
    ax.scatter(locs[:, 1].cpu() * slen - 0.5, 
                locs[:, 0].cpu() * slen - 0.5, 
                marker = marker, color = color)
    
    
