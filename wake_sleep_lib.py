import torch

import numpy as np

import simulated_datasets_lib
import inv_kl_objective_lib as inv_kl_lib
import utils
import image_utils

from psf_transform_lib import get_psf_loss

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_sleep(star_encoder, loader, optimizer, n_epochs, out_filename, iteration):
    print_every = 10

    test_losses = np.zeros((4, n_epochs))

    for epoch in range(n_epochs):
        t0 = time.time()

        # draw fresh data
        loader.dataset.set_params_and_images()

        avg_loss, counter_loss, locs_loss, fluxes_loss \
            = inv_kl_lib.eval_star_encoder_loss(star_encoder, loader,
                                                optimizer, train = True)

        elapsed = time.time() - t0
        print('[{}] loss: {:0.4f}; counter loss: {:0.4f}; locs loss: {:0.4f}; fluxes loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                        epoch, avg_loss, counter_loss, locs_loss, fluxes_loss, elapsed))

        test_losses[:, epoch] = np.array([avg_loss, counter_loss, locs_loss, fluxes_loss])
        np.savetxt(out_filename + '-test_losses-' + 'iter' + str(iteration),
                    test_losses)

        if ((epoch % print_every) == 0) or (epoch == (n_epochs-1)):
            # loader.dataset.set_params_and_images()
            # _ = inv_kl_lib.eval_star_encoder_loss(star_encoder,
            #                                     loader, train = True)
            #
            # loader.dataset.set_params_and_images()
            # test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss = \
            #     inv_kl_lib.eval_star_encoder_loss(star_encoder,
            #                                     loader, train = False)
            #
            # print('**** test loss: {:.3f}; counter loss: {:.3f}; locs loss: {:.3f}; fluxes loss: {:.3f} ****'.format(\
            #     test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss))

            outfile = out_filename + '-iter' + str(iteration)
            print("writing the encoder parameters to " + outfile)
            torch.save(star_encoder.state_dict(), outfile)


# def train_psf_transform_one_epoch(sampled_locs_full_image,
#                                     sampled_fluxes_full_image,
#                                     sampled_n_stars_full,
#                                     log_q_locs, log_q_fluxes, log_q_n_stars,
#                                     psf_transform, optimizer,
#                                     n_samples, batchsize,
#                                     cached_grid = None,
#                                     use_iwae = False):
#
#     avg_loss = 0.0
#     for i in range(n_samples // batchsize):
#         optimizer.zero_grad()
#
#         # get psf
#         psf = psf_transform.forward()
#
#         indx1 = int(i * batchsize)
#         indx2 = min(int((i + 1) * batchsize), n_samples)
#
#         # get loss
#         neg_logprob = get_psf_loss(full_image, full_background,
#                                     sampled_locs_full_image[indx1:indx2],
#                                     sampled_fluxes_full_image[indx1:indx2],
#                                     n_stars = sampled_n_stars_full[indx1:indx2],
#                                     psf = psf,
#                                     pad = 5, grid = cached_grid)[1]
#
#         if use_iwae:
#             # this is log (p / q)
#             log_pq = - neg_logprob - log_q_locs[indx1:indx2] - log_q_fluxes[indx1:indx2] - log_q_n_stars[indx1:indx2]
#             loss_i = - torch.logsumexp(log_pq - np.log(n_samples), 0)
#         else:
#             loss_i = neg_logprob.mean()
#
#         loss_i.backward()
#         optimizer.step()
#
#         avg_loss += loss_i.detach() * (indx2 - indx1) / n_samples
#
#     return avg_loss

def run_wake(full_image, full_background, star_encoder, psf_transform, optimizer,
                n_epochs, n_samples, out_filename, iteration,
                epoch0 = 0,
                use_iwae = False):

    cached_grid = simulated_datasets_lib._get_mgrid(full_image.shape[-1]).to(device).detach()
    print_every = 20

    avg_loss = 0.0
    counter = 0
    t0 = time.time()
    test_losses = []
    for epoch in range(1, n_epochs + 1):

        optimizer.zero_grad()

        # get psf
        psf = psf_transform.forward()

        # get index of sampled parameters
        indx1 = 0 # int((epoch - 1) * n_samples)
        indx2 = n_samples # int(epoch * n_samples)

        sampled_locs_full_image, sampled_fluxes_full_image, sampled_n_stars_full, \
            log_q_locs, log_q_fluxes, log_q_n_stars = \
                star_encoder.sample_star_encoder(full_image, full_background,
                                        n_samples, return_map = False,
                                        return_log_q = use_iwae,
                                        training_fluxes = True)


        # get loss
        neg_logprob = get_psf_loss(full_image, full_background,
                                    sampled_locs_full_image[indx1:indx2].detach(),
                                    sampled_fluxes_full_image[indx1:indx2],
                                    n_stars = sampled_n_stars_full[indx1:indx2].detach(),
                                    psf = psf,
                                    pad = 5, grid = cached_grid)[1]

        if use_iwae:
            # this is log (p / q)
            log_pq = - neg_logprob - log_q_locs[indx1:indx2].detach() - \
                            log_q_fluxes[indx1:indx2].detach() - \
                            log_q_n_stars[indx1:indx2].detach()

            loss_i = - torch.logsumexp(log_pq - np.log(n_samples), 0)
        else:
            loss_i = neg_logprob.mean()

        loss_i.backward()
        optimizer.step()

        avg_loss += loss_i.detach()
        counter += 1

        if ((epoch % print_every) == 0) or (epoch == n_epochs):
            elapsed = time.time() - t0
            print('[{}] loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                        epoch, avg_loss / counter, elapsed))

            test_losses.append(avg_loss / counter)
            np.savetxt(out_filename + '-test_losses-' + 'iter' + str(iteration),
                        test_losses)

            # reset
            avg_loss = 0.0
            counter = 0
            t0 = time.time()

    outfile = out_filename + '-iter' + str(iteration)
    print("writing the psf parameters to " + outfile)
    torch.save(psf_transform.state_dict(), outfile)


def run_joint_wake(full_image, full_background, star_encoder, psf_transform, optimizer,
                    n_epochs, n_samples, out_filename, iteration,
                    epoch0 = 0,
                    use_iwae = False):

    cached_grid = simulated_datasets_lib._get_mgrid(full_image.shape[-1]).to(device).detach()
    print_every = 20

    avg_loss = 0.0
    counter = 0
    t0 = time.time()
    test_losses = []
    for epoch in range(1, n_epochs + 1):

        optimizer.zero_grad()

        # get psf
        psf = psf_transform.forward()

        sampled_locs_full_image, sampled_fluxes_full_image, sampled_n_stars_full, \
            log_q_locs, log_q_fluxes, log_q_n_stars = \
                star_encoder.sample_star_encoder(full_image, full_background,
                                        n_samples,
                                        return_map = False,
                                        return_log_q = True,
                                        training = True)

        # get loss
        neg_logprob = get_psf_loss(full_image, full_background,
                                    sampled_locs_full_image,
                                    sampled_fluxes_full_image,
                                    n_stars = sampled_n_stars_full.detach(),
                                    psf = psf,
                                    pad = 5, grid = cached_grid)[1]

        entropy_term = - log_q_locs - log_q_fluxes - log_q_n_stars

        loss_i = neg_logprob - entropy_term

        ps_loss = loss_i.detach() * log_q_n_stars + loss_i

        ps_loss.mean().backward()
        optimizer.step()

        avg_loss += loss_i.detach().mean()
        counter += 1

        if ((epoch % print_every) == 0) or (epoch == n_epochs):
            elapsed = time.time() - t0
            print('[{}] loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                        epoch, avg_loss / counter, elapsed))

            # save encoder
            outfile = out_filename
            print("writing the encoder parameters to " + outfile)
            torch.save(star_encoder.state_dict(), outfile)

            test_losses.append(avg_loss / counter)
            np.savetxt(out_filename + '-test_losses-' + 'iter' + str(iteration),
                        test_losses)

            # reset
            avg_loss = 0.0
            counter = 0
            t0 = time.time()

    # TODO: save psf
    # outfile = out_filename + '-iter' + str(iteration)
    # print("writing the psf parameters to " + outfile)
    # torch.save(psf_transform.state_dict(), outfile)
