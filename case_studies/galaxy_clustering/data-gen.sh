#!/bin/bash

#echo "Enter number of files to be generated: "
#read n_files

python3 catalog_gen.py "$1"
if [ -z "$1" ]; then
    galsim galsim-des.yaml
else
    galsim galsim-des.yaml variables.nfiles="$1"
fi
