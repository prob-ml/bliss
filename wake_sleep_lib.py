import torch

import simulated_datasets_lib
import inv_kl_objective_lib as inv_kl_lib
from kl_objective_lib import sample_normal, sample_class_weights

from psf_transform_lib import get_psf_transform_loss

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_sleep(star_encoder, loader, optimizer, n_epochs, out_filename, iteration):
    print_every = 10

    test_losses = np.zeros((4, n_epochs // print_every + 1))

    for epoch in range(n_epochs):
        t0 = time.time()

        avg_loss, counter_loss, locs_loss, fluxes_loss \
            = inv_kl_lib.eval_star_encoder_loss(star_encoder, loader,
                                                optimizer, train = True)

        elapsed = time.time() - t0
        print('[{}] loss: {:0.4f}; counter loss: {:0.4f}; locs loss: {:0.4f}; fluxes loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                        epoch, avg_loss, counter_loss, locs_loss, fluxes_loss, elapsed))

        # draw fresh data
        loader.dataset.set_params_and_images()

        if (epoch % print_every) == 0:
            _ = inv_kl_lib.eval_star_encoder_loss(star_encoder,
                                                loader, train = True)

            loader.dataset.set_params_and_images()
            test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss = \
                inv_kl_lib.eval_star_encoder_loss(star_encoder,
                                                loader, train = False)

            print('**** test loss: {:.3f}; counter loss: {:.3f}; locs loss: {:.3f}; fluxes loss: {:.3f} ****'.format(\
                test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss))

            outfile = out_filename + '-iter' + str(iteration)
            print("writing the encoder parameters to " + outfile)
            torch.save(star_encoder.state_dict(), outfile)

            test_losses[:, epoch // print_every] = np.array([test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss])
            np.savetxt(out_filename + '-test_losses-' + 'iter' + str(iteration),
                        test_losses)

def run_wake(full_image, full_background, star_encoder, psf_transform, simulator, optimizer,
                n_epochs, out_filename, iteration):

    for epoch in range(n_epochs):
        t0 = time.time(); print_every = 10

        optimizer.zero_grad()

        # get params: these normally would be the variational parameters.
        # using true parameters atm
        # _, subimage_locs, subimage_fluxes, _, _ = \
        # 	star_encoder.get_image_stamps(full_image, true_full_locs, true_full_fluxes,
        # 									trim_images = False)

        #####################
        # get params
        #####################
        # the image stamps
        image_stamps = star_encoder.get_image_stamps(full_image, locs = None, fluxes = None, trim_images = False)[0]
        background_stamps = star_encoder.get_image_stamps(full_background, locs = None, fluxes = None, trim_images = False)[0]

        # pass through NN
        h = star_encoder._forward_to_last_hidden(image_stamps, background_stamps).detach()
        # get log probs
        log_probs = star_encoder._get_logprobs_from_last_hidden_layer(h)

        # sample number of stars
        n_stars_sampled = sample_class_weights(torch.exp(log_probs))
        is_on_array = simulated_datasets_lib.get_is_on_from_n_stars(n_stars_sampled, star_encoder.max_detections)

        # get variational parameters
        logit_loc_mean, logit_loc_logvar, \
            log_flux_mean, log_flux_logvar = \
                star_encoder._get_params_from_last_hidden_layer(h, n_stars_sampled)

        # sample locations
        subimage_locs_sampled = torch.sigmoid(sample_normal(logit_loc_mean, logit_loc_logvar)) * \
                                    is_on_array.unsqueeze(2).float()

        # sample fluxes
        subimage_fluxes_sampled = torch.exp(sample_normal(log_flux_mean, log_flux_logvar)) * \
                                    is_on_array.float()

        # get loss
        loss = get_psf_transform_loss(full_image, full_background,
                                        subimage_locs_sampled,
                                        subimage_fluxes_sampled,
                                        star_encoder.tile_coords,
                                        star_encoder.stamp_slen,
                                        star_encoder.edge_padding,
                                        simulator,
                                        psf_transform)[1]

        avg_loss = loss.mean()

        avg_loss.backward()
        optimizer.step()

        elapsed = time.time() - t0
        print('[{}] loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                    epoch, avg_loss, elapsed))

        if (epoch % print_every) == 0:
            outfile = out_filename + '-iter' + str(iteration)
            print("writing the psf parameters to " + outfile)
            torch.save(psf_transform.state_dict(), outfile)
