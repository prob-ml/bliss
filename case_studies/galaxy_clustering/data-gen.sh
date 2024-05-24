#!/bin/bash

while getopts ":n:s:t:" opt; do
  case $opt in
    n) nfiles="$OPTARG"
    ;;
    s) image_size="$OPTARG"
    ;;
    t) tile_size="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

echo "Generating Catalogs..."
if [ -z "$nfiles" ]; then
    if [ -z "$image_size" ]; then
        python3 data_generation/catalog_gen.py
    else
        python3 data_generation/catalog_gen.py image_size="$image_size"
    fi
else
    if [ -z "$image_size" ]; then
        python3 data_generation/catalog_gen.py nfiles="$nfiles"
    else
        python3 data_generation/catalog_gen.py nfiles="$nfiles" image_size="$image_size"
    fi
fi
echo "...Done!"
echo "Generating Images..."
if [ -z "$nfiles" ]; then
    if [ -z "$image_size" ]; then
        galsim data_generation/galsim-des.yaml
    else
        galsim data_generation/galsim-des.yaml variables.image_size="$image_size"
    fi
else
    if [ -z "$image_size" ]; then
        galsim data_generation/galsim-des.yaml variables.nfiles="$nfiles"
    else
        galsim data_generation/galsim-des.yaml variables.nfiles="$nfiles" variables.image_size="$image_size"
    fi
fi
echo "...Done!"
echo "Generating File Datums..."
if [ -z "$image_size" ]; then
    if [ -z "$tile_size" ]; then
        python3 data_generation/file_datum_generation.py
    else
        python3 data_generation/file_datum_generation.py tile_size="$tile_size"
    fi
else
    if [ -z "$tile_size" ]; then
        python3 data_generation/file_datum_generation.py image_size="$image_size"
    else
        python3 data_generation/file_datum_generation.py image_size="$image_size" tile_size="$tile_size"
    fi
fi
echo "...Done! Data Generation Complete!"