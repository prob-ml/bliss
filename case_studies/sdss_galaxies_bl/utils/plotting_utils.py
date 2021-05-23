import matplotlib.pyplot as plt

def plot_residuals(image, recon_mean): 
    fig, ax = plt.subplots(1, 3, figsize = (15, 3))
    
    im0 = ax[0].matshow(image[0, 0].cpu())
    fig.colorbar(im0, ax = ax[0])
    
    
    im1 = ax[1].matshow(recon_mean[0, 0].detach().cpu())
    fig.colorbar(im1, ax = ax[1])
    
    resid = (image - recon_mean)[0, 0].detach().cpu()
    vmax = resid.abs().max()
    im2 = ax[2].matshow(resid, 
                        vmax = vmax,
                        vmin = -vmax, 
                        cmap = plt.get_cmap('bwr'))
    fig.colorbar(im2, ax = ax[2])
    
    return fig, ax