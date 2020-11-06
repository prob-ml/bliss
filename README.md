![](http://portal.nersc.gov/project/dasrepo/celeste/sample_sky.jpg)


Bayesian Light Source Separator (BLISS)
========================================
![tests](https://github.com/applied-bayes/bliss/workflows/tests/badge.svg)
[![codecov.io](https://codecov.io/gh/prob-ml/bliss/branch/master/graphs/badge.svg?branch=master&token=Jgzv0gn3rA)](http://codecov.io/github/prob-ml/bliss?branch=master)

BLISS is a Bayesian procedure for deblending light sources. BLISS provides: 
  - __Accurate estimation__ of parameters in blended field.
  - __Calibrated uncertainties__ through fitting an approximate Bayesian posterior.
  - __Scalability__ of Bayesian inference to entire astronomical surveys. 
  
BLISS uses state-of-the-art methods in variational inference including:
  - __Amortized inference__, in which a neural network maps telescope images to an approximate Bayesian posterior on parameters of interest. 
  - __Variational auto-encoders__ (VAEs) to fit a flexible model for galaxy morphology. 
  - __Wake-sleep algorithm__ to jointly fit the approximate posterior and model parameters such as the PSF and the galaxy VAE. 
  
## Latest updates

### Galaxies 
   - BLISS now includes a galaxy model based on a Variational AutoEncoder that was traind on CATSIM bulge+disk galaxies.
   - We are working on testing galaxy detection functionality and developing galaxy shape measurement.
 
### Stars
   - BLISS already includes the StarNet functionality from its predecessor repo: [DeblendingStarFields](https://github.com/Runjing-Liu120/DeblendingStarfields).
  
