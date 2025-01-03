#!/bin/bash
bliss -cp ~/bliss/case_studies/dc2_diffusion -cn train_config
bliss -cp ~/bliss/case_studies/dc2_diffusion -cn slim_train_config
bliss -cp ~/bliss/case_studies/dc2_diffusion -cn half_pixel_train_config
bliss -cp ~/bliss/case_studies/dc2_diffusion -cn only_locs_train_config
