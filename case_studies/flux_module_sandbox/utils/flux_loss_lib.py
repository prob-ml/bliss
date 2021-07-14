import torch
from torch.distributions import normal

from bliss.models.encoder import get_star_bool

def kl_qp_flux_loss(image_decoder,
                        batch, 
                        est_flux, 
                        est_flux_sd): 
    
    batchsize = batch['images'].shape[0]
    assert est_flux.shape == batch['fluxes'].shape
    
    # get reconstruction
    recon, _ = image_decoder.render_images(
                batch["n_sources"],
                batch["locs"],
                batch["galaxy_bool"],
                batch["galaxy_params"],
                est_flux,
                add_noise=False,
            )
        
    # log likelihood
    scale = torch.sqrt(recon.clamp(min = 1.))
    norm = normal.Normal(loc = recon, scale = scale)
    loglik = norm.log_prob(batch['images']).view(batchsize, -1).sum(1)
    
    # entropy
    star_bool = get_star_bool(batch['n_sources'], batch['galaxy_bool'])
    entropy = torch.log(est_flux_sd) * star_bool
    entropy = entropy.view(batchsize, -1).sum(1)
    
    # negative elbo
    kl = - (loglik + entropy)
    
    return kl, -loglik, recon



