#!/bin/bash
set -x
OUTPUT_DIR="./output/train_all"

# if (ls $OUTPUT_DIR > /dev/null 2> /dev/null); then
# 	rm -r $OUTPUT_DIR
# fi
# mkdir $OUTPUT_DIR

if (ls ./data/latents_simulated_sdss_galaxies.pt > /dev/null 2> /dev/null); then
	rm ./data/latents_simulated_sdss_galaxies.pt
fi

# for EXPERIMENT in sdss_autoencoder sdss_binary sdss_galaxy_encoder sdss_sleep
for EXPERIMENT in sdss_galaxy_encoder sdss_sleep
do
	echo "Started $EXPERIMENT..."
	bliss +experiment=$EXPERIMENT training.save_top_k=1 paths.output=$OUTPUT_DIR training.version=$EXPERIMENT training.n_epochs=26
	cp `find $OUTPUT_DIR/default/$EXPERIMENT/checkpoints | tail -n 1` ./models/$EXPERIMENT.ckpt
done