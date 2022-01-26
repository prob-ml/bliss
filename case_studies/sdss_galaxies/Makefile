OUTPUT_DIR:=../output/train_all

all: models/sdss_autoencoder.pt models/sdss_binary.pt models/sdss_galaxy_encoder.pt models/sdss_sleep.pt

models/sdss_autoencoder.pt:
	./run_experiment.sh $(OUTPUT_DIR) sdss_autoencoder
	rm -f ../data/latents_simulated_sdss_galaxies.pt

models/sdss_binary.pt: models/sdss_autoencoder.pt
	./run_experiment.sh $(OUTPUT_DIR) sdss_binary

models/sdss_galaxy_encoder.pt: models/sdss_autoencoder.pt
	./run_experiment.sh $(OUTPUT_DIR) sdss_galaxy_encoder

models/sdss_sleep.pt: models/sdss_autoencoder.pt
	./run_experiment.sh $(OUTPUT_DIR) sdss_sleep

models/sdss_galaxy_encoder_real.pt: models/sdss_sleep.pt models/sdss_autoencoder.pt
	./run_experiment.sh $(OUTPUT_DIR) sdss_galaxy_encoder_real
