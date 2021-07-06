import torch
from torch.distributions import normal

import time

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
    
    x = batch['image']
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