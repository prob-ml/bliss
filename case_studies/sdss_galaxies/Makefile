GPU:=$(CUDA_VISIBLE_DEVICES)
GPU_BINARY:=$(GPU)
GPU_GALAXY_ENCODER:=$(GPU)
GPU_LOCATION_ENCODER:=$(GPU)

all: models/sdss_autoencoder.pt models/sdss_binary.pt models/sdss_galaxy_encoder.pt models/sdss_detection_encoder.pt

models/sdss_autoencoder.pt:
	./run_experiment.sh sdss_autoencoder $(GPU)
	rm -f models/latents_simulated_sdss_galaxies.pt

models/sdss_binary.pt: models/sdss_autoencoder.pt
	./run_experiment.sh sdss_binary $(GPU_BINARY)

models/sdss_galaxy_encoder.pt: models/sdss_autoencoder.pt
	./run_experiment.sh sdss_galaxy_encoder $(GPU_GALAXY_ENCODER)

models/sdss_detection_encoder.pt: models/sdss_autoencoder.pt
	./run_experiment.sh sdss_detection_encoder $(GPU_LOCATION_ENCODER)

clean:
	rm models/sdss_autoencoder.pt models/sdss_binary.pt models/sdss_galaxy_encoder.pt models/sdss_detection_encoder.pt models/latents_simulated_sdss_galaxies.pt
