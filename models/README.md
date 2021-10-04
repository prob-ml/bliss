# Models

This README should be regularly updated to contain the commands used to produce each of the models
in this directory. Please also specify most recent **validation loss** and number of
**epochs** as a reference.

* ``sdss_autoencoder.ckpt``

```bash
# Validation Loss = 777883.625
# Epochs = 2699
# Notes:
# - 2021-10-01: Changed to fully-convolutional dual-autoencoder
poetry run bliss mode="train" model="galaxy_net" dataset="sdss_galaxies" optimizer="adam" \
training.n_epochs=7500 training.trainer.check_val_every_n_epoch=10 dataset.kwargs.noise_factor=0.01 \
dataset.kwargs.num_workers=5 training.trainer.checkpoint_callback=True
```

* ``sdss_binary.ckpt``

```bash
# Validation Loss = 5.420
# Epochs = 379
# Notes:
# - 2021-10-01: Changed to fully-convolutional dual-autoencoder
# - 20/09/21: Slight increase since galaxy PSF exactly equals decoder PSF
poetry run bliss mode="train" model="binary_sdss" dataset="default" \
optimizer.kwargs.lr=1e-4 training.n_epochs=501 training.trainer.checkpoint_callback=True \
model.decoder.kwargs.mean_sources=0.03 training.save_top_k=5
```

* ``sdss_galaxy_encoder.ckpt``

```bash
# Validation Loss = 302512
# Epochs = 3374
# Notes:
# - 2021-10-04: Changed to fully-convolutional dual-autoencoder
# - 20/09/21: Training for even longer might be possible.
poetry run bliss mode="train" model="galaxy_encoder_sdss" dataset="default" \
optimizer="adam" optimizer.kwargs.lr=1e-4 training.n_epochs=5001 training.trainer.checkpoint_callback=True \
model.decoder.kwargs.mean_sources=0.04 training.trainer.check_val_every_n_epoch=25
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
poetry run bliss mode="train" model="sleep_sdss_detection" dataset="default" \
optimizer.kwargs.lr=1e-4 training.n_epochs=1501 training.trainer.checkpoint_callback=True \
model.decoder.kwargs.mean_sources=0.03 training.save_top_k=10
```
