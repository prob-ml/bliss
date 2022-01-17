OUTPUT_DIR:=../output/train_all
GPU:=0
GPU_BINARY:=$(GPU)
GPU_GALAXY_ENCODER:=$(GPU)
GPU_SLEEP:=$(GPU)

all: sdss_autoencoder.ckpt sdss_binary.ckpt sdss_galaxy_encoder.ckpt sdss_sleep.ckpt

sdss_autoencoder.ckpt: 
	./run_experiment.sh $(OUTPUT_DIR) sdss_autoencoder $(GPU)

sdss_autoencoder_flow.ckpt: sdss_autoencoder.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_autoencoder_flow $(GPU)

sdss_binary.ckpt: sdss_autoencoder.ckpt sdss_autoencoder_flow.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_binary $(GPU_BINARY)

sdss_galaxy_encoder.ckpt: sdss_autoencoder.ckpt sdss_autoencoder_flow.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_galaxy_encoder $(GPU_GALAXY_ENCODER)

sdss_sleep.ckpt: sdss_autoencoder.ckpt sdss_autoencoder_flow.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_sleep $(GPU_SLEEP)

sdss_galaxy_encoder_real.ckpt: sdss_sleep.ckpt sdss_autoencoder.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_galaxy_encoder_real $(GPU_SLEEP)
