# Models

This README should be regularly updated to contain the commands used to produce each of the models
in this directory. Please also specify most recent **validation loss** and number of
**epochs** as a reference.

* ``sdss_autoencoder.ckpt``

```bash
# Validation Loss = 782252.688
# Epochs = 999
poetry run bliss +experiment=sdss_autoencoder
```

* ``sdss_binary.ckpt``

```bash
# Validation Loss = 6.571
# Epochs = 359
# Notes:
# - 20/09/21: Slight increase since galaxy PSF exactly equals decoder PSF
poetry run bliss +experiment=sdss_binary
```

* ``sdss_galaxy_encoder.ckpt``

```bash
# Validation Loss = 1295313.875
# Epochs = 1174
# Notes:
# - 20/09/21: Training for even longer might be possible.
poetry run bliss +experiment=sdss_galaxy_encoder
```

* ``sdss_sleep.ckpt``

```bash
# Validation Loss = -0.102
# Epochs = 1349
# Notes:
# - Optimization can be a bit unstable and not always reach < -0.09 level (which seems to be significant cutoff)
# - Results on SDSS Stripe 82 frame vary +- 10% depending on optimization (at least current metrics), might become
# clear if differences are significant once we make metrics as a function of magnitude. Another possibility
# is that the sleep phase data model misspecification?
poetry run bliss +experiment=sdss_sleep
```
