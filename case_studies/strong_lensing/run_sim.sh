#!/bin/bash

# Script to automate running the galsim command with the yaml configuration

# Default Values
num_files=3
num_galaxies=3
img_size=500


while getopts "f:s:g:" opt; do
  case $opt in
    f) num_files="$OPTARG"  # num output images
    ;;
    s) img_size="$OPTARG" # size of output images
    ;;
    g) num_galaxies="$OPTARG" # number of galaxies in final output image
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


# base image config file
GALAXY_CONFIG_FILE="galsim-random.yaml"
# final image config file
IMAGE_CONFIG_FILE="simulate.yaml"


NUM_ITERATIONS=$((num_galaxies+1))

# Set the output directory for base images
OUTPUT_DIR="data/images"

# Output directory for final images
FINAL_OUTPUT_DIR="output_yaml"

# Catalog Directory
CATALOG_DIR="catalogs"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$CATALOG_DIR"
fi

MAG=0.029999
# Create the output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to create output directory $OUTPUT_DIR!"
        exit 1
    fi
fi

# Log file
LOG_FILE="galsim_run.log"

NUM_UNLENSED_IMAGES=$((num_galaxies/2))
NUM_LENSED_IMAGES=$((num_galaxies-NUM_UNLENSED_IMAGES))

echo "Starting data generation..."
echo "Data generation beginning..." &> $LOG_FILE
for i in $(seq 1 $num_files); do
    echo "Starting File $i / $num_files..." &>> $LOG_FILE

    for j in $(seq 1 $NUM_ITERATIONS); do
      echo "Iteration $j..." &>> $LOG_FILE

      # First Python script - Galaxy Parameter Generation
      PYTHON_SCRIPT_1="random_galaxy_gen.py"
      # Second Python script - Lenstronomy Lensing Logic
      PYTHON_SCRIPT_2="bliss_lens.py"


      output=$(python $PYTHON_SCRIPT_1 $img_size)
      params=($output)

      n1=${params[0]}
      half_light_radius=${params[1]}
      flux1=${params[2]}
      n2=${params[3]}
      scale_radius=${params[4]}
      flux2=${params[5]}
      q=${params[6]}
      beta=${params[7]}
      x=${params[8]}
      y=${params[9]}

      galsim "$GALAXY_CONFIG_FILE" variables.output_directory="$OUTPUT_DIR" variables.n1="$n1" variables.half_light_radius="$half_light_radius" variables.flux1="$flux1" variables.n2="$n2" variables.scale_radius="$scale_radius" variables.flux2="$flux2" variables.q="$q" variables.beta="$beta degrees" &>> $LOG_FILE

      # Check if galsim command was successful
      if [ $? -eq 0 ]; then
          echo "GalSim ran successfully for file $i iteration $j. Output files are in $OUTPUT_DIR." &>> $LOG_FILE
      else
          echo "GalSim encountered an error during file $i iteration $j. Check the log file $LOG_FILE for details." &>> $LOG_FILE
          exit 1
      fi

      # Find the generated .fits file
      FITS_FILE=$(find "$OUTPUT_DIR" -name "galsim.fits" -print -quit)
      if [ -z "$FITS_FILE" ]; then
          echo "No .fits file found in $OUTPUT_DIR for file $i iteration $j!" &>> $LOG_FILE
          exit 1
      fi

      # Write unlensed images to combined files
      if [ "$j" -le "$NUM_UNLENSED_IMAGES" ]; then
          echo "$j, 0, 0, 0, 'F814W', 0, 'combined_images.fits', 'real_galaxy_PSF_images.fits', $j, $j, $MAG, 0, 1.33700996e-05, 'acs_I_unrot_sci_20_cf.fits', 0, False, $x, $y, $n1, $half_light_radius, $flux1, $n2, $scale_radius, $flux2, $q, $beta" >> data/catalog.txt

      # Write lensed images to combined files
      else
          # Running Lenstronomy lensing (Generating lensed image)
          echo "Running Python script $PYTHON_SCRIPT_2 on iteration $j..." &>> $LOG_FILE
          output=$(python "$PYTHON_SCRIPT_2" "$FITS_FILE" "$OUTPUT_DIR")
          lens_params=($output)
          theta_E=${lens_params[0]}
          center_x=${lens_params[1]}
          center_y=${lens_params[2]}
          e1=${lens_params[3]}
          e2=${lens_params[4]}
        #   echo "$ITR, 0, 0, 0, 'F814W', 0, 'combined_images.fits', 'real_galaxy_PSF_images.fits', $ITR, $ITR, $MAG, 0, 1.33700996e-05, 'acs_I_unrot_sci_20_cf.fits', 0, True, $x, $y, $n1, $half_light_radius, $flux1, $n2, $scale_radius, $flux2, $q, $beta, $theta_E, $center_x, $center_y, $e1, $e2" >> data/catalog.txt
          echo "$j, 0, 0, 0, 'F814W', 0, 'combined_images.fits', 'real_galaxy_PSF_images.fits', $j, $j, $MAG, 0, 1.33700996e-05, 'acs_I_unrot_sci_20_cf.fits', 0, True, $x, $y, $n1, $half_light_radius, $flux1, $n2, $scale_radius, $flux2, $q, $beta, $theta_E, $center_x, $center_y, $e1, $e2" >> data/catalog.txt
      fi

      # Rename the output files to include the iteration number
      mv "$OUTPUT_DIR/galsim.fits" "$OUTPUT_DIR/galsim_iter${j}.fits"

    done

    # Third Python script - Stacking Images
    PYTHON_SCRIPT_3="stack_images.py"
    # Check if Python scripts exist
    if [ ! -f "$PYTHON_SCRIPT_3" ]; then
        echo "Python script $PYTHON_SCRIPT_3 not found!"
        exit 1
    fi

    echo "Running python file $PYTHON_SCRIPT_3 to stack all images..." &>> $LOG_FILE
    python "$PYTHON_SCRIPT_3" "$OUTPUT_DIR" &>> $LOG_FILE

    # Fourht Python script - Converting Text Catalog to Fits
    PYTHON_SCRIPT_4="convert_catalog.py"
    # Check if Python scripts exist
    if [ ! -f "$PYTHON_SCRIPT_4" ]; then
        echo "Python script $PYTHON_SCRIPT_4 not found!"
        exit 1
    fi

    echo "Running python file $PYTHON_SCRIPT_4 to convert text catalog..." &>> $LOG_FILE
    python "$PYTHON_SCRIPT_4" "$OUTPUT_DIR" &>> $LOG_FILE

    # Run Galsim to produce final image
    echo "Running galsim with $IMAGE_CONFIG_FILE..." &>> $LOG_FILE
    galsim "$IMAGE_CONFIG_FILE" variables.output_dir="$FINAL_OUTPUT_DIR" variables.nobjects="$num_galaxies" variables.image_size="$img_size" &>> $LOG_FILE

    mv "$FINAL_OUTPUT_DIR/image.fits" "$FINAL_OUTPUT_DIR/image${i}.fits"

    OPEN_FITS="open_fits.py"
    python "$OPEN_FITS" "$i" "$FINAL_OUTPUT_DIR" &>> $LOG_FILE

    rm -r data/images
    rm data/catalog.fits data/combined_images.fits
    sed -i '$d' data/catalog.txt
    mv data/catalog.txt $CATALOG_DIR/image${i}.txt
done

# Move all image pngs to separate folder
mkdir $FINAL_OUTPUT_DIR/images
mkdir $FINAL_OUTPUT_DIR/data

mv $FINAL_OUTPUT_DIR/*.png $FINAL_OUTPUT_DIR/images/
mv $FINAL_OUTPUT_DIR/*.fits $FINAL_OUTPUT_DIR/data/

echo "Data generation completed."
echo "Data generation ended."&>> $LOG_FILE
