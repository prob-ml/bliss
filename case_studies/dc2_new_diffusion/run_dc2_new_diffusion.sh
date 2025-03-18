#!/bin/bash
bliss -cp ~/bliss/case_studies/dc2_new_diffusion/upsample_diffusion_config -cn upsample_diffusion_train_config
bliss -cp ~/bliss/case_studies/dc2_new_diffusion/latent_diffusion_config -cn latent_diffusion_train_config
bliss -cp ~/bliss/case_studies/dc2_new_diffusion/ynet_diffusion_config -cn ynet_diffusion_train_config
bliss -cp ~/bliss/case_studies/dc2_new_diffusion/ynet_dd_diffusion_config -cn ynet_dd_diffusion_train_config
bliss -cp ~/bliss/case_studies/dc2_new_diffusion/ynet_diffusion_config -cn ynet_full_diffusion_train_config
