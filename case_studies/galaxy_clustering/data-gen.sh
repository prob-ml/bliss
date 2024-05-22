#!/bin/bash

#echo "Enter number of files to be generated: "
#read n_files

echo "Generating Catalogs..."
python3 data_generation/catalog_gen.py "$1"
echo "...Done!"
echo "Generating Images..."
if [ -z "$1" ]; then
    galsim data_generation/galsim-des.yaml
else
    galsim data_generation/galsim-des.yaml variables.nfiles="$1"
fi
echo "...Done!"
echo "Generating File Datums..."
python3 data_generation/file_datum_generation.py
echo "...Done! Data Generation Complete!"
