OUTPUT_DIR:=../output/train_all

all: sdss_autoencoder.pt sdss_binary.pt sdss_galaxy_encoder.pt sdss_sleep.pt

sdss_autoencoder.pt:
	./run_experiment.sh $(OUTPUT_DIR) sdss_autoencoder
	rm -f ../data/latents_simulated_sdss_galaxies.pt

sdss_binary.pt: sdss_autoencoder.pt
	./run_experiment.sh $(OUTPUT_DIR) sdss_binary

sdss_galaxy_encoder.pt: sdss_autoencoder.pt
	./run_experiment.sh $(OUTPUT_DIR) sdss_galaxy_encoder

sdss_sleep.pt: sdss_autoencoder.pt
	./run_experiment.sh $(OUTPUT_DIR) sdss_sleep

sdss_galaxy_encoder_real.pt: sdss_sleep.pt sdss_autoencoder.pt
	./run_experiment.sh $(OUTPUT_DIR) sdss_galaxy_encoder_real
