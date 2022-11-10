all: models/galaxy_encoder.pt models/lens_encoder.pt

models/galaxy_encoder.pt: 
	./run_experiment.sh galaxy_encoder

models/lens_encoder.pt: 
	./run_experiment.sh lens_encoder

clean:
	rm models/binary.pt models/galaxy_encoder.pt models/detection_encoder.pt models/latents_simulated_galaxies.pt models/lensing_binary_encoder.pt models/lensed_galaxy_encoder.pt models/lens_encoder.pt
