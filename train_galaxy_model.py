#!/usr/bin/env python3

import argparse
import numpy as np
import subprocess
import timeit
from pathlib import Path

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, sampler
import torch.optim as optim

import datasets
import galaxy_net


torch.backends.cudnn.benchmark = True
plt.switch_backend("Agg")


parser = argparse.ArgumentParser(description='GalaxyNet',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training.')
parser.add_argument('--mode', type=str, default="rnn", metavar='N',
                    help='specifies the dataset and the vae type: one, rnn, or grid')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train.')
parser.add_argument('--seed', type=int, default=64, metavar='S',
                    help='random seed.')
parser.add_argument('--nocuda', action='store_true',
                    help="whether to using a discrete graphics card")
parser.add_argument('--device', type=int, default=0, metavar='N',
                    help='GPU device ID')
parser.add_argument('--dir', type=str, default="data/test", metavar='N',
                    help='run-specific directory to read from / write to')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite if directory already exists.')
args = parser.parse_args()


torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


def plot_reconstruction(data_loader, vae, epoch, sleep=False):
    num_examples = min(10, args.batch_size)

    plt.ioff()
    plt.figure(figsize=(5 * (3), 2 + 4 * num_examples))
    plt.tight_layout()
    plt.suptitle("Epoch {:d}".format(epoch))

    num_cols = 3 #also look at recon_var 

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            image = data["image"].cuda() #copies from cpu to gpu memory. 
            background = data["background"].cuda() #maybe not having background will be a problem. 
            num_galaxies = data["num_galaxies"]
            vae.eval()

            if sleep: #what does this part do? 
                really_is_on, _, _, image = vae.synthetic_image(background)
                num_galaxies = really_is_on.sum(dim=0).cpu().int().detach().numpy()

            recon_mean, recon_var, _ = vae(image, background)

            for i in range(num_examples):
                vmax1 = image[i, 2].max() #we are looking at the ith sample in the second band. 
                plt.subplot(num_examples, num_cols, num_cols * i + 1)
                plt.title("image [{} galaxies]".format(num_galaxies[i]))
                plt.imshow(image[i, 2].data.cpu().numpy(), vmax=vmax1)
                plt.colorbar()

                plt.subplot(num_examples, num_cols, num_cols * i + 2)
                plt.title("recon_mean")
                plt.imshow(recon_mean[i, 2].data.cpu().numpy(), vmax=vmax1)
                plt.colorbar()

                plt.subplot(num_examples, num_cols, num_cols * i + 3)
                plt.title("recon_var")
                plt.imshow(recon_var[i, 2].data.cpu().numpy(), vmax=vmax1)
                plt.colorbar()

            break

    plots = Path(args.dir, 'sleep_plots' if sleep else 'plots')
    plots.mkdir(parents=True, exist_ok=True)
    plot_file = plots.joinpath("plot_epoch_{}".format(epoch))
    plt.savefig(plot_file.as_posix())
    plt.close()


def train_epoch(vae, train_loader, optimizer):
    vae.train() #set in training mode. 
    avg_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        image = data["image"].cuda()
        background = data["background"].cuda()

        loss = vae.loss(image, background)
        avg_loss += loss.item()

        optimizer.zero_grad() #clears the gradients of all optimized torch.Tensors
        loss.backward()
        optimizer.step()

    avg_loss /= len(train_loader.sampler)
    return avg_loss


def eval_epoch(vae, data_loader):
    vae.eval() #set in evaluation mode. 
    avg_loss = 0.0

    with torch.no_grad(): #no need to compute gradients outside training. 
        for batch_idx, data in enumerate(data_loader):
            image = data["image"].cuda() #shape: [nsamples, num_bands, slen, slen]
            background = data["background"].cuda()
            loss = vae.loss(image, background)
            avg_loss += loss.item()

    avg_loss /= len(data_loader.sampler)
    return avg_loss


def train_module(vae, ds, epochs=10000, lr=1e-4):
    optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-6)

    tt_split = int(0.1 * len(ds)) #10% of data only. 
    test_indices = np.mgrid[:tt_split]
    train_indices = np.mgrid[tt_split:len(ds)]

    test_loader = DataLoader(ds, batch_size=args.batch_size,
                             num_workers=2, pin_memory=True,
                             sampler=sampler.SubsetRandomSampler(test_indices))
    train_loader = DataLoader(ds, batch_size=args.batch_size,
                              num_workers=2, pin_memory=True,
                              sampler=sampler.SubsetRandomSampler(train_indices))

    for epoch in range(0, epochs):
        np.random.seed(args.seed + epoch)
        start_time = timeit.default_timer()
        batch_loss = train_epoch(vae, train_loader, optimizer)
        elapsed = timeit.default_timer() - start_time
        print('[{}] loss: {:.3f}  \t[{:.1f} seconds]'.format(epoch, batch_loss, elapsed))

        if epoch % 10 == 0:
            print("  * plotting reconstructions...")
            plot_reconstruction(test_loader, vae, epoch)

            print("  * writing the network's parameters to disk...")
            params = Path(args.dir, 'params')
            params.mkdir(parents=True, exist_ok=True)
            vae_file = params.joinpath("vae_params_{}.dat".format(epoch))
            dec_file = params.joinpath("dec_params_{}.dat".format(epoch))
            torch.save(vae.state_dict(), vae_file.as_posix())
            torch.save(vae.dec.state_dict(), dec_file.as_posix())

            print("  * evaluating test loss...")
            test_loss = eval_epoch(vae, test_loader)
            print("  * test loss: {:.0f}\n".format(test_loss))
            loss_file = Path(args.dir, "loss.dat")
            with open(loss_file.as_posix(), 'a') as f: 
                f.write(f"epoch {epoch}, test loss: {test_loss}\n")


def run():
    """ 
    Questions: 
    * What does vae.cuda() do? 
    """
    ds = datasets.Synthetic(15, min_galaxies=1, max_galaxies=1, mean_galaxies=1, num_images=12800, centered=True)
    vae = galaxy_net.OneCenteredGalaxy(15)
    vae.cuda()

    print("training...")
    train_module(vae, ds, epochs=args.epochs)


if __name__ == "__main__":
    #check if directory exists
    if Path(args.dir).is_dir() and not args.overwrite:
        raise IOError("Directory already exists.")

    elif Path(args.dir).is_dir(): 
        subprocess.run(f"rm -r {args.dir}", shell=True)


    with torch.cuda.device(args.device):
        run()
