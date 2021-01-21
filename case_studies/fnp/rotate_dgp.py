import torch
import torch.distributions as dist
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

def make_rot_matrices(phis):
    cos_phi = phis.cos()
    sin_phi = phis.sin()
    row1    = torch.stack([cos_phi, -sin_phi], -1)
    row2    = torch.stack([sin_phi, cos_phi], -1)
    y       = torch.stack([row1, row2], -1)
    return y

def make_X_ref_pair(x1, x2, K):
    X = x1 + (x2 - x1)/(K+1) * (torch.tensor(range(K), dtype=torch.float32) + 1.0)
    return X

# TODO: Add jitter here
def make_X(n_ref, Ks):
    X_ref = torch.tensor(range(0, n_ref*2, 2), dtype=torch.float32)
    X_dep = torch.tensor([], dtype=torch.float32)
    for i in range(n_ref - 1):
        X_dep = torch.cat([X_dep, make_X_ref_pair(X_ref[i], X_ref[i+1], Ks[i])])
    X_all, idxs = torch.cat([X_ref, X_dep], 0).sort()

    Is = torch.tensor(range(X_all.size(0)))
    idx_ref = Is[idxs < n_ref]
    idx_dep = Is[idxs >= n_ref]
    return X_ref, X_dep, X_all, idx_ref, idx_dep

def make_graph_ref_pair(j1, j2, K, n_ref):
    """
        This calculates dependency matrix for points between j1 and j2
        :param j1: Index of first reference point
        :param j2: Index of second reference point
        :param K: Number of dependent points
        :param n_ref: Number of reference_points
    """
    X = torch.zeros(K, n_ref)
    X[:, j1] = 1
    X[:, j2] = 1
    return X
 
def make_graphs(n_ref, Ks): 
    """
        This calculates the dependency matrices G and A, assuming that there are Ks[j] interpolated points between 
        references j and j+1
        :param n_ref: Number of reference_points
        :param Ks: List of integers indicated number of interpolated points in that gap
    """
    G   =  torch.zeros(n_ref, n_ref, dtype=torch.float32)
    for j in range(n_ref-1):
        G[j+1, j] = 1.
    
    A   = torch.tensor([], dtype=torch.float32)
    for j in range(n_ref-1):
        A = torch.cat([A, make_graph_ref_pair(j, j+1, Ks[j], n_ref)])
    
    return G, A

class PSFRotate:
    """
    Class for generating synthetic sequences of stars which rotate.
    """
    def __init__(self, X, size_h=10, size_w=10, base_angle=math.pi/100, angle_stdev=math.pi/300,
            cov_multiplier = 6.0, bright_val=3.0, bright_skip=5,
            star_width=0.25):
        """
            :param X: Locations of the stars
            :param size_h: The height of each image in pixels
            :param size_w: The width of each image in pixels
            :param base_angle: How much the star rotates on average in radians per unit of X
            :param angle_stdev: The standard deviation of rotation (flat; not per unit of X!)
            :param cov_multiplier: How much covariance the log normal density serving as the PSF will have
            :param bright_val: How much brighter are bright stars
            :param bright_skip: Which stars are bright
            :param star_width: The width of the star (lower values lead to skinnier stars)
        """

        self.X              = X
        self.size_h         = size_h
        self.size_w         = size_w
        self.base_angle     = base_angle
        self.angle_stdev    = angle_stdev
        self.cov_multiplier = cov_multiplier
        self.bright_val     = bright_val
        self.bright_skip    = bright_skip
        self.star_width     = star_width

        self.base_cov       = torch.tensor([[1.0, 0.0], [0.0, self.star_width]]) * self.cov_multiplier
        self.I              = self.X.size(0)

        self.h    = torch.tensor(range(self.size_h), dtype=torch.float32)
        self.w    = torch.tensor(range(self.size_w), dtype=torch.float32)
        self.grid = torch.stack([self.h.unsqueeze(1).repeat(1, self.size_w), self.w.unsqueeze(0).repeat(self.size_h, 1)], 2)

    def generate(self, N, device=None):
        start   = torch.rand(N, device=device) * math.pi
        eps     = torch.randn(N, self.I, device=device) * self.angle_stdev
        phi     = start.unsqueeze(1) + (self.X.to(device).unsqueeze(0) + eps) * (self.base_angle)

        idx_brights    = torch.fmod(torch.tensor(range(self.I), device=device), self.bright_skip) == 0
        l              = torch.ones(N, self.I, device=device)
        l[:, idx_brights] = self.bright_val
        
        mu         = torch.tensor([torch.mean(self.h), torch.mean(self.w)], device=device)
        rots       = make_rot_matrices(phi)
        covs       = rots.transpose(3,2).matmul(self.base_cov.to(device)).matmul(rots).unsqueeze(2).unsqueeze(2)
        pixel_dist = dist.MultivariateNormal(mu, covs)

        brights    = pixel_dist.log_prob(self.grid.to(device).unsqueeze(0)).exp() * l.unsqueeze(-1).unsqueeze(-1)
        return brights
    
class PsfFnpData:
    def __init__(self, n_ref, Ks, N, N_valid=None, conv=False, device=None, **kwargs):
        self.n_ref  = n_ref
        self.Ks     = Ks
        self.N      = N
        self.conv   = conv
        self.device = device

        if N_valid is None:
            self.N_valid = self.N
        else:
            self.N_valid = N_valid

        self.X_ref, self.X_dep, self.X_all, self.idx_ref, self.idx_dep = make_X(n_ref, Ks)
        self.G, self.A = make_graphs(n_ref, Ks)

        self.dgp     = PSFRotate(self.X_all, **kwargs)

        self.images, self.stdx, self.stdy, X, y = self.generate(self.N)
        self.X_r, self.y_r, self.X_m, self.y_m, self.X, self.y = self.split_reference_dependent(X, y)

        self.images_valid, _, _, X, y = self.generate(self.N_valid, self.stdx, self.stdy)
        self.X_r_valid, self.y_r_valid, self.X_m_valid, self.y_m_valid, self.X_valid, self.y_valid = self.split_reference_dependent(X, y)

    def generate(self, N, stdx=None, stdy=None):
        images    = self.dgp.generate(N, device=self.device).cpu()
        Xmat      = self.X_all.unsqueeze(1)
        ymat      = images.reshape(N, self.dgp.I, -1)

        if stdx is None:
            stdx = StandardScaler().fit(Xmat)
        if stdy is None:
            stdy = StandardScaler().fit(ymat.reshape(N*self.dgp.I, -1))
        X, y       = stdx.transform(Xmat), stdy.transform(ymat.reshape(N*self.dgp.I, -1)).reshape(N, self.dgp.I, -1)

        return images, stdx, stdy, X, y
    
    def split_reference_dependent(self, X, y):
        idxR = self.idx_ref
        idxM = self.idx_dep
        N    = y.shape[0]

        X_r = torch.from_numpy(X[idxR, :].astype(np.float32))
        y_r = torch.from_numpy(y[:, idxR, :].astype(np.float32))
        X_m = torch.from_numpy(X[idxM, :].astype(np.float32))
        y_m = torch.from_numpy(y[:, idxM, :].astype(np.float32))
        X   = torch.from_numpy(X.astype(np.float32))
        y   = torch.from_numpy(y.astype(np.float32))

        if self.conv:
            y_r      = y_r.reshape(N, self.n_ref, 1, self.dgp.size_h, self.dgp.size_w)
            y_m      = y_m.reshape(N, self.dgp.I - self.n_ref, 1, self.dgp.size_h, self.dgp.size_w)
            y        = y.reshape(N, self.dgp.I, 1, self.dgp.size_h, self.dgp.size_w)

        return X_r, y_r, X_m, y_m, X, y
    
    def markref(self, img, max_bright=None):
        if max_bright is None:
            max_bright = self.images.max()
        img[0, :]                   = max_bright
        img[self.dgp.size_h - 1, :] = max_bright
        img[:, 0]                   = max_bright
        img[:, self.dgp.size_w - 1] = max_bright
        return img

    def export_images(self, path, mark_ref=True, valid=False, nrows=None):
        if valid:
            images = self.images_valid
        else: 
            images = self.images
        if nrows is None:
            nrows = self.N
        vmin = self.images.min()
        vmax = self.images.max()
        image_lng   = torch.tensor([])
        for n in range(nrows):
            row = torch.tensor([])
            for i in range(self.dgp.I):
                img = images[n, i]
                if mark_ref and (i in self.idx_ref):
                    img = self.markref(img)
                row = torch.cat([row, img], dim=1)
            image_lng = torch.cat([image_lng, row], dim=0)
        plt.imsave(path, image_lng, vmin=vmin, vmax=vmax)
        
    def cuda(self):
        self.X_r, self.X_m, self.X                   = self.X_r.cuda(), self.X_m.cuda(), self.X.cuda()
        self.y_r, self.y_m, self.y                   = self.y_r.cuda(), self.y_m.cuda(), self.y.cuda()
        self.y_r_valid, self.y_m_valid, self.y_valid = self.y_r_valid.cuda(), self.y_m_valid.cuda(), self.y_valid.cuda()

    def cpu(self):
        self.X_r, self.X_m, self.X = self.X_r.cpu(), self.X_m.cpu(), self.X.cpu()
        self.y_r, self.y_m, self.y = self.y_r.cpu(), self.y_m.cpu(), self.y.cpu()
    
    def predict_n(self, y_r, fnp_model, X=None, A=None, sample_Z=True):
        """
        Make a prediction using the generating X_dep
        :param n: The index of the Ys to use
        :param fnp_model: Trained RegressionFNP
        """
        if X is None:
            X = self.X_m
        if A is None:
            A = self.A.cuda()
        pred_np = fnp_model.predict(X, self.X_r, y_r, A_in=A, sample_Z=sample_Z)
        # pred    = torch.from_numpy(pred_np)
        # newsize = torch.Size([pred.size(0), self.dgp.size_h, self.dgp.size_w])
        return pred_np[0]
    
    def quantiles_n(self, y_r, fnp_model, quantiles=[0.05, 0.95], samples=1000):
        preds = []
        for i in range(samples):
            preds.append(self.predict_n(y_r, fnp_model))
        pred_tens = torch.stack(preds)
        quant_out = []
        for q in quantiles:
            quant_out.append(torch.from_numpy(np.percentile(pred_tens, q, axis=0)).to(torch.float32))
        
        return quant_out
    
    def mean_n(self, y_r, fnp_model, X=None, A=None, samples=1000, sample_Z=True):
        preds = []
        for i in range(samples):
            preds.append(self.predict_n(y_r, fnp_model, X=X, A=A, sample_Z=sample_Z))
        pred_tens = torch.stack(preds)
        return pred_tens.mean(dim=0)

    def make_fnp_pred_image_n(self, X_dep, Y_d, n, valid=True):
        if valid:
            images = self.images_valid
        else:
            images = self.images
        X_all, idxs = torch.cat([self.X_ref, X_dep]).sort()
        Is          = torch.tensor(range(X_all.size(0)))
        idxs_ref    = Is[idxs < self.n_ref]
        idxs_dep    = Is[idxs >= self.n_ref]

        max_bright  = torch.max(Y_d)

        img         = torch.tensor([])

        for i in Is:
            if i in idxs_ref:
                img_i = self.markref(images[n, i, :, :])
            else:
                img_i = Y_d[idxs_dep == i, :].squeeze(0)
            img = torch.cat([img, img_i], dim = 1)
        
        return img

    def make_fnp_pred_image(self, fnp_model):
        imgs  = []
        for i in range(self.N):
            predi = self.predict_n(i, fnp_model)
            imgi  = self.make_fnp_pred_image_n(self.X_dep, predi, i)
            imgs.append(imgi)

        myimg = torch.cat(imgs, 0)
        return myimg
    
    def make_fnp_quant_image(self, fnp_model, Ns=None, quantiles=[0.05, 0.95], samples=1000, valid=True):
        if Ns is None:
            Ns = range(self.N)

        quants = []
        for i in Ns:
            quanti = self.quantiles_n(i, fnp_model, quantiles=quantiles, samples=samples)
            quants.append(quanti)
        
        quantimgs = []
        for j in range(len(quantiles)):
            imgs = []
            for i in Ns:
                predi = quants[i][j]
                imgi  = self.make_fnp_pred_image_n(self.X_dep, predi, i, valid=valid)
                imgs.append(imgi)
            my_img = torch.cat(imgs, 0)
            quantimgs.append(my_img)

        return quantimgs

    def make_fnp_mean_image(self, fnp_model, X=None, X_nostd=None, A=None, N=None, samples=1000, valid=True, sample_Z=True):
        imgs  = []
        if N is None:
            if valid:
                N = self.N_valid
            else:
                N = self.N
        for i in range(N): 
            if valid:
                y_r = self.y_r_valid[i:(i+1)]
            else:
                y_r = self.y_r[i:(i+1)]
            predi = self.mean_n(y_r, fnp_model, X=X, A=A, samples=samples, sample_Z=sample_Z)
            if X_nostd is None:
                X_dep = self.X_dep.cpu()
            else:
                X_dep = X_nostd.cpu()
            imgi  = self.make_fnp_pred_image_n(X_dep, predi, i, valid=valid)
            imgs.append(imgi)

        myimg = torch.cat(imgs, 0)
        return myimg

    def make_fnp_single_image(self, fnp_model, N=None, valid=True, sample_Z=True):
        return self.make_fnp_mean_image(fnp_model, N=N, samples=1, valid=valid, sample_Z=sample_Z)
    def make_fnp_var_image(self, fnp_model, N=None, samples=1000, valid=True):
        imgs  = []
        if N is None:
            if valid:
                N = self.N_valid
            else:
                N = self.N
        for i in range(N): 
            if valid:
                y_r = self.y_r_valid[i:(i+1)]
            else:
                y_r = self.y_r[i:(i+1)]
            predi = self.mean_n(y_r.pow(2), fnp_model, samples=samples) - self.mean_n(y_r, fnp_model, samples=samples).pow(2)
            imgi  = self.make_fnp_pred_image_n(self.X_dep, predi, i, valid=valid)
            imgs.append(imgi)

        myimg = torch.cat(imgs, 0)
        return myimg
