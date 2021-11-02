OUTPUT_DIR:=../output/train_all

all: sdss_autoencoder.ckpt sdss_binary.ckpt sdss_galaxy_encoder.ckpt sdss_sleep.ckpt

sdss_autoencoder.ckpt: 
	./run_experiment.sh $(OUTPUT_DIR) sdss_autoencoder
	rm -f ../data/latents_simulated_sdss_galaxies.pt

sdss_binary.ckpt: sdss_autoencoder.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_binary

sdss_galaxy_encoder.ckpt: sdss_autoencoder.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_galaxy_encoder

sdss_sleep.ckpt: sdss_autoencoder.ckpt sdss_galaxy_encoder.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_sleep
