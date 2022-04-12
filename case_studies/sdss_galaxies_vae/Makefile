GPU:=0
GPU_BINARY:=$(GPU)
GPU_GALAXY_ENCODER:=$(GPU)
GPU_SLEEP:=$(GPU)

all: models/sdss_vae.pt models/sdss_binary.pt models/sdss_galaxy_encoder.pt models/sdss_location_encoder.pt models/sdss_galaxy_encoder_real.pt

models/sdss_vae.pt:
	./run_experiment.sh sdss_vae $(GPU)

models/sdss_vae_flow.pt: models/sdss_vae.pt
	./run_experiment.sh sdss_vae_flow $(GPU)

models/sdss_binary.pt: models/sdss_vae.pt
	./run_experiment.sh sdss_binary $(GPU_BINARY)

models/sdss_galaxy_encoder.pt: models/sdss_vae.pt
	./run_experiment.sh sdss_galaxy_encoder $(GPU)

models/sdss_location_encoder.pt: models/sdss_vae.pt
	./run_experiment.sh sdss_location_encoder $(GPU_SLEEP)

models/sdss_galaxy_encoder_real.pt: models/sdss_location_encoder.pt models/sdss_binary.pt models/sdss_vae.pt models/sdss_galaxy_encoder.pt
	./run_experiment.sh sdss_galaxy_encoder_real $(GPU_SLEEP)
