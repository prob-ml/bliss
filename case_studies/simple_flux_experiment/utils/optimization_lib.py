import torch
from torch.distributions import normal

import time

from which_device import device

#####################
# several candidate loss functions
#####################

def l2_loss(network, batch): 
    
    x = batch['image']
    y = batch['flux'].squeeze()
        
    est, sd = network(x)
    
    loss = (y - est)**2
            
    return loss, y, est, sd


def l2_loss_logspace(network, batch): 
    
    images = batch['image'] 
    
    x = torch.log(images - images.min() + 1.0)
    y = torch.log(batch['flux'].squeeze())
    
    est, sd = network(x)
    
    loss = (y - est)**2
    
    return loss, y, est, sd

def klpq(network, batch): 
    
    x = batch['image']
    y = batch['flux'].squeeze()
        
    est, sd = network(x)
        
    norm = normal.Normal(loc = est, scale = sd)
    
    loss = - norm.log_prob(y)
    
    return loss, y, est, sd

def klqp(network, batch, psf, background): 
    
    x = batch['image']
    y = batch['flux'].squeeze()
    
    batchsize = x.shape[0]
    
    assert psf.shape[0] == x.shape[2]
    assert psf.shape[1] == x.shape[3]
    slen = psf.shape[0]
    
    # get estimate
    est, sd = network(x)
    
    # sample from variational distribution
    z = torch.randn(est.shape, device = device)
    sample = est + z * sd

    # reconstruct image 
    recon = psf.unsqueeze(0) * sample.view(batchsize, 1, 1) + background
    
    # add in the one band
    recon = recon.view(batchsize, 1, slen, slen)
    
    # log likelihood
    scale = torch.sqrt(recon.clamp(min = 1.))
    norm = normal.Normal(loc = recon, scale = scale)
    loglik = norm.log_prob(x).view(batchsize, -1).sum(1)
    
    # entropy
    entropy = torch.log(sd)
    
    # negative elbo
    loss = - (loglik + entropy)
    
    return loss, y, est, sd


#####################
# wrapper to train the network
#####################
def train_network(network, loss_fun, dataset, optimizer, n_epochs): 
    network.train();
    t0 = time.time() 

    for epoch in range(n_epochs): 

        avg_loss = 0.

        for _, batch in enumerate(dataset):

            optimizer.zero_grad()

            loss = loss_fun(network, batch)[0].mean()
            loss.backward()

            optimizer.step()

            avg_loss += loss 

        print('epoch [{}]. loss = {}'.format(epoch, avg_loss / len(dataset)))
    
    network.eval();
    print('done. Elapsed {:.03f}sec'.format(time.time() - t0))