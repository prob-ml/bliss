## DC2 Diffusion Experiment Record

### 01-12

1. use the new diffusion model with objective as "pred_x0" ("sigmoid" beta schedule)
2. use the new diffusion model with objective as "pred_noise" ("sigmoid" beta schedule)
3. use the old diffusion model
4. use the new diffusion model with objective as "pred_noise" and use "linear" beta schedule
5. use the new diffusion model with objective as "pred_x0" and use "linear" beta schedule
6. use the new diffusion model ("pred_x0" and "sigmoid" schedule); use new detection net with ori norm layer
7. use the new diffusion model ("pred_x0" and "sigmoid" schedule); use new detection net with group norm layer

### 01-13

1. use the new diffusion model ("pred_x0" and "sigmoid" schedule); use new detection net with group norm layer and new time embedding
2. use the new diffusion model ("pred_x0" and "sigmoid" schedule); use new detection net with group norm layer and new time embedding (don't predict flux and ellip) (**hereafter, we use this as default setting**)

### 01-15

1. set ddim_steps to 100, without self cond
2. set ddim_steps to 100, with self cond
3. half pixel, new network and weighted n_sources loss (weight=100)
4. half pixel, new network and weighted n_sources loss (weight=10)
5. half pixel, new network and weighted n_sources loss (weight=5)

### 01-17

1. slim encoder with SimpleDetectionNet
2. slim encoder with SimpleDetectionNet; remove feature pre-process and set num of features to 64
3. slim encoder with SimpleDetectionNet; remove feature pre-process and set num of features to 32
4. slim encoder with DetectionNet; use catalog_parser to decode and encode x_start (**hereafter, we use this as default setting**)

### 01-18

1. set ddim_steps to 100
2. set ddim_steps to 200
3. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 100; pred_x0
4. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 100; pred_noise
5. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 100; pred_noise; use multiple bits (enable individual_hw and set one_side_parts to 4) for locs; lr scheduler has milestones [30, 60, 90], decay is set to 0.3
6. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 100; pred_x0; use multiple bits (enable individual_hw and set one_side_parts to 4) for locs; lr scheduler has milestones [30, 60, 90], decay is set to 0.3; set bit_value to 1.0

### 01-19

(find that more ddim_steps will degrade F1 score)
1. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 2.0
2. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 2.0; fill empty tile with random noise
3. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 2.0; fill empty tile with random noise; set n_sources threshold to 0.5

4. with SimpleDetectionNet, remove the feats_scale_shift; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 2.0; set n_sources threshold to 0.0
5. with SimpleDetectionNet, remove the feats_scale_shift; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 2.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear
6. with SimpleDetectionNet, remove the feats_scale_shift; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 2.0; set n_sources threshold to 0.0; set ddim_beta_schedule to cosine
7. with SimpleDetectionNet, remove the feats_scale_shift; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 1.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear


(fix the config parameters bug)
8. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 1.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear; add fake tiles
9. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 1.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear; fill empty tile with random noise
10. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 1.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear; add fake tiles with n_sources' fake_data_on_prop = 0.05
11. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 1.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear; add fake tiles with n_sources' fake_data_on_prop = 1.0


(fix the time emb bug)
12. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 1.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear; add fake tiles with n_sources' fake_data_on_prop = 1.0
13. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 1.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear; add fake tiles with n_sources' fake_data_on_prop = 0.5
14. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 1.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear; add fake tiles with n_sources' fake_data_on_prop = 0.05
15. with SimpleDetectionNet; set num of features to 128; set ddim_steps to 5; pred_x0; lr scheduler has milestones [30, 60, 90], decay is set to 0.1; set scale to 1.0; set n_sources threshold to 0.0; set ddim_beta_schedule to linear; fill empty tiles with noise


### 01-20

1. test only locs diffusion; this diffusion is based on DiffusionDet; set locs scale to 2.0
2. test only locs diffusion; this diffusion is based on DiffusionDet; set locs scale to 1.0


### 01-23

1. use upsampled target catalog as input for the feature net.