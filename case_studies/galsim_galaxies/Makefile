GPU:=$(CUDA_VISIBLE_DEVICES)
GPU_BINARY:=$(GPU)
GPU_GALAXY_ENCODER:=$(GPU)
GPU_LOCATION_ENCODER:=$(GPU)

all: models/autoencoder.pt models/binary_encoder.pt models/galaxy_encoder.pt models/detection_encoder.pt

models/autoencoder.pt:
	./run_experiment.sh autoencoder $(GPU)

models/detection_encoder.pt: models/autoencoder.pt
	./run_experiment.sh detection_encoder $(GPU_LOCATION_ENCODER)

models/binary_encoder.pt: models/autoencoder.pt
	./run_experiment.sh binary_encoder $(GPU_BINARY)

models/galaxy_encoder.pt: models/autoencoder.pt
	./run_experiment.sh galaxy_encoder $(GPU_GALAXY_ENCODER)

clean:
	rm models/autoencoder.pt models/binary_encoder.pt models/galaxy_encoder.pt models/detection_encoder.pt
