#!/bin/bash

# Usage: generate_data_in_parallel.sh <relative config_path> <num_files_per_process> <num_processes> <num_workers_per_process>
CONFIG_PATH=$1
N_FILES_PER_PROCESS=$2
NUM_PROCESSES=${3:-32}
N_WORKERS_PER_PROCESS=${4:-0}

# Gracefully exit when interrupted
pids=()
cleanup() {
    echo "Ending background processes"
    for pid in "${pids[@]}";
    do
        kill -15 $pid
    done
}
trap cleanup SIGINT

# Error if CONFIG_PATH, N_FILES_PER_PROCESS not specified
if [ -z "$CONFIG_PATH" ] || [ -z "$N_FILES_PER_PROCESS" ]; then
    echo "Usage: generate_data_in_parallel.sh <relative config_path> <num_files_per_process> <num_processes (default: 32)> <num_workers_per_process (default: 0)>"
    exit 1
fi

for ((i=0; i<$NUM_PROCESSES; i++));
do
    files_start_idx=$((i * N_FILES_PER_PROCESS))
    bliss -cp $CONFIG_PATH 'mode=generate' "simulator.num_workers=$N_WORKERS_PER_PROCESS" "generate.files_start_idx=$files_start_idx" &
    pids+=($!) # store PID in array
done
wait
