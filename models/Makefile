OUTPUT_DIR:=../output/train_all

all: sdss_autoencoder.ckpt sdss_binary.ckpt sdss_galaxy_encoder.ckpt sdss_sleep.ckpt

sdss_autoencoder.ckpt: 
	./run_experiment.sh $(OUTPUT_DIR) sdss_autoencoder

sdss_autoencoder_flow.ckpt: sdss_autoencoder.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_autoencoder_flow.ckpt

sdss_binary.ckpt: sdss_autoencoder.ckpt sdss_autoencoder_flow.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_binary

sdss_galaxy_encoder.ckpt: sdss_autoencoder.ckpt sdss_autoencoder_flow.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_galaxy_encoder

sdss_sleep.ckpt: sdss_autoencoder.ckpt sdss_autoencoder_flow.ckpt
	./run_experiment.sh $(OUTPUT_DIR) sdss_sleep
