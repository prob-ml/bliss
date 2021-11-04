# Models
-  All checkpoints were generated using `make all`. See the Makefile for details.
-  Results from the latest run for each `$EXPERIMENT` are in `results_$EXPERIMENT`.
-  This README should be regularly updated with any notes on significant changes to model training or performance.


* ``sdss_autoencoder.ckpt``

* ``sdss_binary.ckpt``

```bash
# Notes:
# - 2021-10-01: Changed to fully-convolutional dual-autoencoder
# - 20/09/21: Slight increase since galaxy PSF exactly equals decoder PSF
```

* ``sdss_galaxy_encoder.ckpt``

```bash
# Notes:
# - Training for even longer than 1000 epochs might be possible.
```

* ``sdss_sleep.ckpt``

```bash
# Notes:
# - Optimization can be a bit unstable and not always reach < -0.09 level (which seems to be significant cutoff)
# - Results on SDSS Stripe 82 frame vary +- 10% depending on optimization (at least current metrics), might become
# clear if differences are significant once we make metrics as a function of magnitude. Another possibility
# is that the sleep phase data model misspecification?
# 10/05/21: Slight decrease not significant as long as < -0.09
```
