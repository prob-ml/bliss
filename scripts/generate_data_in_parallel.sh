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

# parse command line arguments
CONFIG_OVERRIDES=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--num-processes)
      NUM_PROCESSES="$2"
      shift;shift;;
    -cp|--config-path)
      CONFIG_PATH="$2"
      shift;shift;;
    -cn|--config-name)
      CONFIG_NAME="$2"
      shift;shift;;
    -*|--*)
      echo "Unknown option $1"
      exit 1;;
    *)
      CONFIG_OVERRIDES="$CONFIG_OVERRIDES $1"
      shift;;
  esac
done


# Error if NUM_PROCESS not specified
if [ -z "$NUM_PROCESSES" ]; then
    echo "Usage: generate_data_in_parallel.sh -n NUM_PROCESSES [-cp CONFIG_PATH] [-cn CONFIG_NAME] [CONFIG_OVERRIDES]"
    exit 1
fi

pids=()
for ((i=0; i<$NUM_PROCESSES; i++));
do
    bliss \
        ${CONFIG_PATH:+-cp $CONFIG_PATH} \
        ${CONFIG_NAME:+-cn $CONFIG_NAME} \
        'mode=generate' \
        "+generate.process_index=$i" \
        $CONFIG_OVERRIDES &
    pids+=($!) # store PID in array
done

# wait for all bliss processes to finish
for pid in "${pids[@]}"; do
    wait $pid
done

