import time


#####################
# several candidate loss functions
#####################

def l2_loss(network, batch): 
    
    images = batch['image']
    fluxes = batch['flux'].squeeze()
        
    mean, _ = network(images)
            
    return (fluxes - mean)**2


def klpq(network, batch): 
    
    images = batch['image']
    fluxes = batch['flux'].squeeze()
        
    mean, sd = network(images)
        
    norm = normal.Normal(loc = mean, scale = sd)
    
    return - norm.log_prob(fluxes)

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

            loss = loss_fun(network, batch).mean()
            loss.backward()

            optimizer.step()

            avg_loss += loss 

        print('epoch [{}]. loss = {}'.format(epoch, avg_loss / len(dataset)))
    
    network.eval();
    print('done. Elapsed {:.03f}sec'.format(time.time() - t0))