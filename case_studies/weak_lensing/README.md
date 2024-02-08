# Weak Lensing
### Winter 2024
#### Undergrads: Tahseen Younus, Steve Fan
#### Supervisors: Jeffrey Regier (faculty), Tim White (PhD student)

## Description
Dark matter between Earth and distant galaxies acts as a “gravitational lens,” distorting
the appearance of these galaxies (see figure). By analyzing images of galaxies, we can
learn about the spatial distribution of dark matter throughout the universe. In this project,
we take a probabilistic approach to mapping dark matter: given the images (observed
random variables), we infer the distribution of dark matter (latent random variables) under
a scientifically plausible generative model. To perform inference, we use a new technique
called neural posterior estimation, which involves simulating many astronomical images
with various dark matter distributions, and then training a convolutional neural network to
predict the location of dark matter for each image. Undergraduate researchers will help
develop the image simulator (making use of existing software), implement several
varieties of neural posterior estimation, and apply them to real astronomical images. No
prior knowledge of astronomy is expected. Familiarity with Bayesian statistics is helpful
but not essential. Strong computational skills are required.

## References
- [Variational Inference for Deblending Crowded Starfields](https://arxiv.org/pdf/2102.02409.pdf)

## Documentation
- TBD

## TODOs
- Fill in lensing_metrics.py
- Implement realistic priors for shear and convergence
- Research traditional methods of convergence estimation — see Kaiser-Squires method and Wiener filter
- Research traditional methods of shear estimation

## URPS
This project is being conducted under the Undergraduate Research Program in Statistics (URPS), a competitive program that pairs promising undergraduates with Statistics faculty on a research project for the winter semester. If you are interested, and for more information, please follow this [link](https://lsa.umich.edu/stats/undergraduate-students/undergraduate-research-opportunities-.html).
