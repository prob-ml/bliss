import pytest

import torch
import bliss.models.fnp as fnp


class TestFNP:
    def test_fnp(self, paths):
        # One dimensional example
        od = fnp.OneDimDataset()

        vanilla_fnp = fnp.RegressionFNP(
            dim_x=1,
            dim_y=1,
            transf_y=od.stdy,
            dim_h=100,
            dim_u=3,
            n_layers=1,
            num_M=od.XM.size(0),
            dim_z=50,
            fb_z=1.0,
            use_plus=False,
        )

        fnpp = fnp.RegressionFNP(
            dim_x=1,
            dim_y=1,
            transf_y=od.stdy,
            dim_h=100,
            dim_u=3,
            n_layers=1,
            num_M=od.XM.size(0),
            dim_z=50,
            fb_z=1.0,
            use_plus=True,
        )

        attt = fnp.PoolingFNP(
            dim_x=1,
            dim_y=1,
            transf_y=None,
            dim_h=100,
            dim_u=3,
            n_layers=1,
            num_M=od.XM.size(0),
            dim_z=50,
            fb_z=1.0,
            use_plus=False,
            use_direction_mu_nu=True,
            set_transformer=True,
        )

        poolnp = fnp.PoolingFNP(
            dim_x=1,
            dim_y=1,
            transf_y=od.stdy,
            dim_h=100,
            dim_u=3,
            n_layers=1,
            num_M=od.XM.size(0),
            dim_z=50,
            fb_z=0.5,
            use_plus=False,
            use_direction_mu_nu=True,
            set_transformer=False,
        )

        model_names = ["fnp", "fnp plus", "pool - attention", "pool - deep set"]
        thresholds = [0.5, 0.6, 0.5, 0.6]

        for (i, model) in enumerate([vanilla_fnp, fnpp, attt, poolnp]):
            print(model_names[i])
            if torch.cuda.is_available():
                od.cuda()
                model = model.cuda()
            model, loss_initial, loss_final = fnp.train_onedim_model(
                model, od, epochs=1000, lr=1e-4
            )
            assert loss_final < loss_initial * thresholds[i]
