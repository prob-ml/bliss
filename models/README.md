# Models

This README should be regularly updated to contain the commands used to produce each of the models
in this directory. Please also specify most recent **validation loss** and number of
**epochs** as a reference.

* ``sdss_autoencoder.ckpt``

```bash
# Validation Loss = 782776.500
# Epochs = 959
poetry run bliss mode="train" model="galaxy_net" dataset="sdss_galaxies" optimizer="adam" \
training.n_epochs=1001 training.trainer.check_val_every_n_epoch=10 dataset.kwargs.noise_factor=0.01 \
dataset.kwargs.num_workers=5 training.trainer.checkpoint_callback=True
```

* ``sdss_binary.ckpt``

```bash
# Validation Loss = 6.571
# Epochs = 359
# Notes:
# - 20/09/21: Slight increase since galaxy PSF exactly equals decoder PSF
poetry run bliss mode="train" model="binary_sdss" dataset="default" \
optimizer.kwargs.lr=1e-4 training.n_epochs=1001 training.trainer.checkpoint_callback=True \
model.decoder.kwargs.mean_sources=0.03 training.save_top_k=5
```

* ``sdss_galaxy_encoder.ckpt``

```bash
# Validation Loss = 1295313.875
# Epochs = 1174
# Notes:
# - 20/09/21: Training for even longer might be possible.
poetry run bliss mode="train" model="galenc_sdss" dataset="default" \
optimizer.kwargs.lr=1e-4 training.n_epochs=1501 training.trainer.checkpoint_callback=True \
model.kwargs.decoder_kwargs.mean_sources=0.04 training.trainer.check_val_every_n_epoch=25
```

* ``sdss_sleep.ckpt``

```bash
# Validation Loss = -0.094
# Epochs = 899
# Notes:
# - Optimization can be a bit unstable and not always reach < -0.09 level (which seems to be significant cutoff)
# - Results on SDSS Stripe 82 frame vary +- 10% depending on optimization (at least current metrics), might become
# clear if differences are signifcant once we make metrics as a function of magnitude.
poetry run bliss mode="train" model="sleep_sdss_detection" dataset="default" \
optimizer.kwargs.lr=1e-4 training.n_epochs=1001 training.trainer.checkpoint_callback=True \
model.decoder.kwargs.mean_sources=0.03 training.save_top_k=5
```
