# Models
NOTE! Dummy checkpoint values used for testing.

This README should be regularly updated to contain the commands used to produce each of the models
in this directory. Please also specify most recent **validation loss** and number of
**epochs** as a reference.

All checkpoints were generated with `./scripts/train_all.sh`.


* ``sdss_autoencoder.ckpt``

```bash
# Validation Loss = 782377.312 
# Epochs = 939
```

* ``sdss_binary.ckpt``

```bash
# Validation Loss = 6.953
# Epochs = 459
# Notes:
# - 2021-10-01: Changed to fully-convolutional dual-autoencoder
# - 20/09/21: Slight increase since galaxy PSF exactly equals decoder PSF
```

* ``sdss_galaxy_encoder.ckpt``

```bash
# Validation Loss = 12952888
# Epochs = 999
# Notes:
# - Training for even longer than 1000 epochs might be possible.
```

* ``sdss_sleep.ckpt``

```bash
# Validation Loss = -0.093
# Epochs = 1129
# Notes:
# - 2021-10-05: Changed to fully-convolutional dual-autoencoder
# - Optimization can be a bit unstable and not always reach < -0.09 level (which seems to be significant cutoff)
# - Results on SDSS Stripe 82 frame vary +- 10% depending on optimization (at least current metrics), might become
# clear if differences are significant once we make metrics as a function of magnitude. Another possibility
# is that the sleep phase data model misspecification?
# 10/05/21: Slight decrease not significant as long as < -0.09
```
