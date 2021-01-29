import pytest

import torch
import bliss.models.fnp as fnp
import os
import math
from torch.optim import Adam


class TestFNP:
    def test_fnp_onedim(self, paths):
        # One dimensional example
        od = fnp.OneDimDataset()

        vanilla_fnp = fnp.RegressionFNP(
            dim_x=1,
            dim_y=1,
            transf_y=od.stdy,
            dim_h=100,
            dim_u=3,
            n_layers=1,
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
            dim_z=50,
            fb_z=0.5,
            use_plus=False,
            use_direction_mu_nu=True,
            set_transformer=False,
        )

        model_names = ["fnp", "fnp plus", "pool - attention", "pool - deep set"]
        thresholds = [0.5, 0.75, 0.5, 0.75]

        for (i, model) in enumerate([vanilla_fnp, fnpp, attt, poolnp]):
            print(model_names[i])
            if torch.cuda.is_available():
                od.cuda()
                model = model.cuda()
            model, loss_initial, loss_final, loss_best = fnp.train_onedim_model(
                model, od, epochs=1000, lr=1e-4
            )
            assert loss_best < loss_initial * thresholds[i]
            # Smoke test for prediction
            pred = model.predict(od.XM, od.XR, od.yR[0].unsqueeze(0))

    def test_fnp_rotate(self, paths):
        ## Rotating star

        if torch.get_num_threads() > 8:
            torch.set_num_threads(8)

        # print("Setting CUDA device to", args.cuda_device[0])
        # torch.cuda.set_device(args.cuda_device[0])

        epochs_outer = 100
        # generate = True
        n_ref = 10
        N = 10
        size_h = 50
        size_w = 50
        cov_multiplier = 20.0
        base_angle = math.pi * (3.0 / 23.0)
        angle_stdev = math.pi * 0.025

        K_probs = torch.tensor([0.25] * 4)
        Ks = torch.multinomial(K_probs, n_ref - 1, replacement=True) + 1

        batch_size = 1
        learnrate = 0.001

        print("Generating data...")
        star_data = fnp.PsfFnpData(
            n_ref,
            Ks,
            N,
            size_h=size_h,
            size_w=size_w,
            cov_multiplier=cov_multiplier,
            base_angle=base_angle,
            angle_stdev=angle_stdev,
            bright_skip=1,
            star_width=0.1,
            conv=True,
            N_valid=10,
            device="cpu",
        )
        star_data.export_images("TEST_rotating_star_dgp.png", nrows=10)
        star_data.export_images(
            "TEST_rotating_star_dgp_valid.png", nrows=10, valid=True
        )
        os.remove("TEST_rotating_star_dgp.png")
        os.remove("TEST_rotating_star_dgp_valid.png")

        fnp_model = fnp.ConvPoolingFNP(
            dim_x=1,
            size_h=size_h,
            size_w=size_w,
            kernel_sizes=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            conv_channels=[16, 32, 64, 128, 256],
            transf_y=star_data.stdy,
            dim_h=100,
            dim_u=1,
            pooling_layers=[32, 16],
            n_layers=3,
            dim_z=8,
            fb_z=1.0,
            use_plus=False,
            G=star_data.G,
            A=star_data.A,
            mu_nu_layers=[128, 64, 32],
            use_x_mu_nu=False,
            use_direction_mu_nu=True,
            output_layers=[32, 64, 128],
            x_as_u=True,
            discrete_orientation=False,
            set_transformer=True,
        )

        if torch.cuda.is_available():
            star_data.cuda()
            fnp_model = fnp_model.cuda()

        optimizer = Adam(fnp_model.parameters(), lr=learnrate)
        fnp_model.train()

        loss_initial = (
            fnp_model(
                star_data.X_r_valid,
                star_data.y_r_valid[[0]],
                star_data.X_m_valid,
                star_data.y_m_valid[[0]],
            )
            / star_data.N_valid
        )
        y_r_in = star_data.y_r_valid[[2]]
        y_m_in = star_data.y_m_valid[[2]]
        loss_initial = (
            fnp_model(star_data.X_r_valid, y_r_in, star_data.X_m_valid, y_m_in)
            / batch_size
        )
        print("Initial loss : {:.3f}".format(loss_initial.item()))
        callback = 100
        epochs = epochs_outer * star_data.N
        nbatches = star_data.N // batch_size
        for i in range(epochs_outer):
            points = torch.randperm(star_data.N)
            for i_n in range(nbatches):
                ns = points[(batch_size * i_n) : (batch_size * (i_n + 1))]
                optimizer.zero_grad()
                y_r_in = star_data.y_r[ns]
                y_m_in = star_data.y_m[ns]
                loss = (
                    fnp_model(star_data.X_r, y_r_in, star_data.X_m, y_m_in) / batch_size
                )
                loss.backward()
                optimizer.step()
                if ((i * N) + i_n) % int(epochs / callback) == 0:
                    print(
                        "({:05.1f}%)Epoch {}/{}. loss: {:.3f}".format(
                            ((i * N) + i_n) / epochs * 100,
                            (i * N) + i_n,
                            epochs,
                            loss.item(),
                        )
                    )

        loss_final = (
            fnp_model(
                star_data.X_r_valid,
                star_data.y_r_valid[[0]],
                star_data.X_m_valid,
                star_data.y_m_valid[[0]],
            )
            / star_data.N_valid
        )
        print("Final loss : {:.3f}".format(loss_final.item()))
        assert loss_final < loss_initial - 5000
