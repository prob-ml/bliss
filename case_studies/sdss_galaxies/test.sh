#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DIR=$SCRIPT_DIR
for i in {1..2}
do
   DIR="$(dirname "$DIR")"
   echo $DIR
done
