# Models

This README should be regularly updated to contain the commands used to produce each of the models
in this directory.

* ``sdss_autoencoder.ckpt``

```bash
poetry run bliss mode="train" model="galaxy_net" dataset="sdss_galaxies" optimizer="adam" \
training.n_epochs=251 training.trainer.check_val_every_n_epoch=10 dataset.kwargs.noise_factor=0.01 \
dataset.kwargs.num_workers=5 training.trainer.checkpoint_callback=True
```

* ``sdss_binary.ckpt``

```bash
poetry run bliss mode="train" model="binary_sdss" dataset="default" \
optimizer.kwargs.lr=1e-4 training.n_epochs=1001 training.trainer.checkpoint_callback=True \
model.decoder.kwargs.mean_sources=0.03 training.save_top_k=5
```

* ``sdss_galaxy_encoder.ckpt``

```bash
poetry run bliss mode="train" model="galenc_sdss" dataset="default" \
optimizer.kwargs.lr=1e-4 training.n_epochs=1001 training.trainer.checkpoint_callback=True \
model.kwargs.decoder_kwargs.mean_sources=0.04 training.trainer.check_val_every_n_epoch=25
```

* ``sdss_sleep.ckpt``

```bash
poetry run bliss mode="train" model="sleep_sdss_detection" dataset="default" \
optimizer.kwargs.lr=1e-4 training.n_epochs=1001 training.trainer.checkpoint_callback=True \
model.decoder.kwargs.mean_sources=0.03 training.save_top_k=5
```
