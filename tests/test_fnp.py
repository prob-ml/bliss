import numpy as np
from pytorch_lightning import LightningModule, Trainer

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam

from bliss.models.fnp import (
    DepGraph,
    FNP,
    AveragePooler,
    SetPooler,
    RepEncoder,
)

from bliss.utils import MLP, SequentialVarg, SplitLayer, ConcatLayer, NormalEncoder


class TestFNP:
    def test_fnp_onedim(self, devices):
        # One dimensional example
        od = OneDimDataset()
        vanilla_fnp = OneDimFNP()
        fnpp = OneDimFNP(use_plus=True)
        attt = OneDimFNP(use_set_pooler=True, use_attention=True)
        poolnp = OneDimFNP(use_set_pooler=True, use_attention=False, fb_z=0.5)
        model_names = ["fnp", "fnp plus", "pool - attention", "pool - deep set"]
        for (i, model) in enumerate([vanilla_fnp, fnpp, attt, poolnp]):
            print(model_names[i])
            train_loader = DataLoader(
                [[od.XR, od.yR, od.XM, od.yM]], batch_size=None, batch_sampler=None
            )
            val_loader = DataLoader(
                [[od.XR, od.yhR, od.XM, od.yhM]], batch_size=None, batch_sampler=None
            )
            trainer = Trainer(
                gpus=devices.gpus,
                max_epochs=1000 if devices.use_cuda else 2,
                logger=None,
                check_val_every_n_epoch=100 if devices.use_cuda else 1,
                checkpoint_callback=False,
            )
            trainer.fit(model, train_loader, val_loader)
            assert model.valid_losses[0] < 70
            assert model.valid_losses[0] > 40
            if devices.use_cuda:
                thresholds = [0.5, 0.75, 0.5, 0.75]
                assert min(model.valid_losses) < model.valid_losses[0] * thresholds[i]
            # Smoke test for prediction
            model.fnp.predict(od.XM, od.XR, od.yR[0].unsqueeze(0))
            model.fnp.predict(
                od.XM, od.XR, od.yR[0].unsqueeze(0), sample=False, sample_Z=False
            )


## Onedim example
class OneDimDataset:
    def __init__(
        self,
        N=20,
        num_extra=500,
        seed=1,
    ):
        ## Generate the first row as in the FNP paper
        np.random.seed(seed)
        X = np.concatenate(
            [
                np.random.uniform(low=0, high=0.6, size=(N - 8, 1)),
                np.random.uniform(low=0.8, high=1.0, size=(8, 1)),
            ],
            axis=0,
        )
        eps = np.random.normal(0.0, 0.03, size=(X.shape[0], 1))
        self.f = lambda x, eps: x + np.sin(4 * (x + eps)) + np.sin(13 * (x + eps)) + eps
        y = self.f(X, eps)
        ## Generate more y-values
        ys = [y]
        for _ in range(99):
            Xi = X + np.random.normal()
            eps_i = np.random.normal(0.0, 0.03, size=(X.shape[0], 1))
            yi = self.f(Xi, eps_i)
            ys.append(yi)
        y = np.concatenate(ys, axis=1).transpose()
        ## Generate holdouts
        ys = []
        for i in range(10):
            Xi = X + np.random.normal()
            eps_i = np.random.normal(0.0, 0.03, size=(X.shape[0], 1))
            yi = self.f(Xi, eps_i)
            ys.append(yi)
        yh = np.concatenate(ys, axis=1).transpose()
        idx = np.arange(X.shape[0])
        ## Pick which indicies are reference points
        self.idxR = np.array([2, 16, 9, 6, 17, 12, 4, 15, 1, 14])
        self.idxM = np.array([i for i in idx if i not in self.idxR.tolist()])

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(2)
        self.XR, self.yR = self.X[self.idxR], self.y[:, self.idxR]
        self.XM, self.yM = self.X[self.idxM], self.y[:, self.idxM]

        ## Holdouts
        yh = torch.from_numpy(yh.astype(np.float32))
        self.yh = yh.unsqueeze(2)
        self.yhR = self.yh[:, self.idxR]
        self.yhM = self.yh[:, self.idxM]

        ## Point where predictions will be made for plotting
        self.dx = np.linspace(-1.0, 2.0, num_extra).astype(np.float32)[:, np.newaxis]

    def cuda(self):
        for nm in ["XR", "XM", "X", "yR", "yM", "y", "yhR", "yhM", "yh"]:
            setattr(self, nm, getattr(self, nm).cuda())

    def cpu(self):
        for nm in ["XR", "XM", "X", "yR", "yM", "y", "yhR", "yhM", "yh"]:
            setattr(self, nm, getattr(self, nm).cpu())


class OneDimFNP(LightningModule):
    def __init__(
        self,
        dim_x=1,
        dim_y=1,
        dim_z=50,
        dim_u=3,
        dim_y_enc=100,
        fb_z=1.0,
        use_plus=False,
        use_set_pooler=False,
        pooling_rep_size=32,
        pooling_layers=(64,),
        use_attention=False,
        st_numheads=(2, 2),
        lr=1e-4,
    ):
        super().__init__()
        cov_vencoder = SequentialVarg(
            MLP(dim_x, [100], 2 * dim_u),
            SplitLayer(dim_u, -1),
            NormalEncoder(),
        )
        dep_graph = DepGraph(dim_u)
        trans_cond_y = MLP(dim_y, [128], dim_y_enc)
        if use_set_pooler:
            rep_encoder = RepEncoder(
                MLP(dim_y_enc + dim_u, [128], pooling_rep_size),
                use_u_diff=True,
                use_x=False,
            )
            pooler = SequentialVarg(
                SetPooler(
                    pooling_rep_size,
                    dim_z,
                    pooling_layers,
                    use_attention,
                    st_numheads,
                ),
                SplitLayer(dim_z, -1),
                NormalEncoder(minscale=1e-8),
            )
        else:
            rep_encoder = RepEncoder(
                MLP(dim_y_enc + dim_x, [128], 2 * dim_z), use_u_diff=False, use_x=True
            )
            pooler = SequentialVarg(
                AveragePooler(dim_z),
                SplitLayer(dim_z, -1),
                NormalEncoder(minscale=1e-8),
            )
        prop_vencoder = SequentialVarg(
            ConcatLayer([1, 0]),
            MLP(
                dim_y_enc + dim_x,
                [128],
                2 * dim_z,
            ),
            SplitLayer(dim_z, -1),
            NormalEncoder(minscale=1e-8),
        )
        output_in = [0] if not use_plus else [0, 1]
        output_insize = dim_z if not use_plus else dim_z + dim_u
        label_vdecoder = SequentialVarg(
            ConcatLayer(output_in),
            MLP(output_insize, [128], 2 * dim_y),
            SplitLayer(dim_y, -1),
            NormalEncoder(minscale=0.1),
        )
        self.fnp = FNP(
            cov_vencoder,
            dep_graph,
            trans_cond_y,
            rep_encoder,
            pooler,
            prop_vencoder,
            label_vdecoder,
            fb_z=fb_z,
        )
        self.lr = lr
        self.valid_losses = []

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        XR, yR, XM, yM = batch
        loss = self.fnp(XR, yR, XM, yM)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.valid_losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
