#!/bin/bash


# Gracefully exit when interrupted
cleanup() {
    echo "Ending background processes"
    for pid in "${pids[@]}";
    do
        kill -15 $pid
    done
}

# gracefully exit on these conditions
trap "cleanup" SIGINT SIGTERM

NUM_PROCESSES=${1:-""}
CONFIG_OVERRIDES=${2:-""}

# Error if NUM_PROCESS not specified
if [ -z "$NUM_PROCESSES" ]; then
    echo "Usage: generate_data_in_parallel.sh <num_processes> [<config_overrides>]"
    exit 1
fi

pids=()
for ((i=0; i<$NUM_PROCESSES; i++));
do
    bliss 'mode=generate' "generate.process_index=$i" $CONFIG_OVERRIDES &
    pids+=($!) # store PID in array
done

# wait for all bliss processes to finish
for pid in "${pids[@]}"; do
    wait $pid
done

