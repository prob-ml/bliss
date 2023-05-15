#!/bin/bash

NUM_PROCESSES=${1:-32}
NUM_WORKERS_PER_PROCESS=${2:-0}
CONFIG_PATH=${3:-"case_studies/summer_template"}

for ((i=0; i<$NUM_PROCESSES; i++));
do
    bliss -cp $CONFIG_PATH 'mode=generate' 'generate.splits=[train]' "simulator.num_workers=$NUM_WORKERS_PER_PROCESS" "generate.train_file_prefix=dataset_p$i" &
done
wait
