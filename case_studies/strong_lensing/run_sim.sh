#!/bin/bash

# Script to automate running the galsim command with the yaml configuration

num_iterations=10
num_files=10
img_size=2000

while getopts "i:f:s:" opt; do
  case $opt in
    i) num_iterations="$OPTARG" # num base/lensed images
    ;;
    f) num_files="$OPTARG"  # num output images
    ;;
    s) img_size="$OPTARG" # size of output images
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

# Set the configuration file
GALAXY_CONFIG_FILE="galsim-random.yaml"

# Number of iterations
# NUM_ITERATIONS=10

# Set the output directory
OUTPUT_DIR="data/images"

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

# First Python script
PYTHON_SCRIPT_1="bliss-lens.py"


# Check if Python scripts exist
if [ ! -f "$PYTHON_SCRIPT_1" ]; then
    echo "Python script $PYTHON_SCRIPT_1 not found!"
    exit 1
fi

echo "Running galsim with $GALAXY_CONFIG_FILE to generate $num_iterations base and strong lensed images..."
# Run the pipeline for each iteration
for i in $(seq 1 $num_iterations); do
    echo "Iteration $i..."

    # Run the galsim command (Generating base image)
    
    galsim "$GALAXY_CONFIG_FILE" variables.output_directory="$OUTPUT_DIR" &> "$LOG_FILE"

    # Check if galsim command was successful
    if [ $? -eq 0 ]; then
        echo "GalSim ran successfully for iteration $i. Output files are in $OUTPUT_DIR."
    else
        echo "GalSim encountered an error during iteration $i. Check the log file $LOG_FILE for details."
        exit 1
    fi

    # Find the generated .fits file
    FITS_FILE=$(find "$OUTPUT_DIR" -name "galsim.fits" -print -quit)

    if [ -z "$FITS_FILE" ]; then
        echo "No .fits file found in $OUTPUT_DIR for iteration $i!"
        exit 1
    fi

    # Running Lenstronomy lensing (Generating lensed image)
    echo "Running Python script $PYTHON_SCRIPT_1 with $FITS_FILE..."
    python "$PYTHON_SCRIPT_1" "$FITS_FILE" "$OUTPUT_DIR" "$OUTPUT_DIR/lenstronomy_iter${i}.png"

    # Check if the first Python script was successful
    if [ $? -eq 0 ]; then
        echo "Python script $PYTHON_SCRIPT_1 ran successfully for iteration $i.\n"
    else
        echo "Python script $PYTHON_SCRIPT_1 encountered an error during iteration $i."
        exit 1
    fi


    # Rename the output files to include the iteration number
    mv "$OUTPUT_DIR/galsim.fits" "$OUTPUT_DIR/galsim_iter${i}.fits"
    mv "$OUTPUT_DIR/lensed_image.fits" "$OUTPUT_DIR/lensed_iter${i}.fits"
    mv "$OUTPUT_DIR/galsim_epsf.fits" "$OUTPUT_DIR/galsim_epsf_iter${i}.fits"

    echo "Renamed files to correct iteration..."

done

# Stacking images Python script
PYTHON_SCRIPT_2="stack_images.py"
# Check if Python scripts exist
if [ ! -f "$PYTHON_SCRIPT_2" ]; then
    echo "Python script $PYTHON_SCRIPT_2 not found!"
    exit 1
fi

echo "Running python file $PYTHON_SCRIPT_2 to stack all images..."
python "$PYTHON_SCRIPT_2" "$OUTPUT_DIR" &> $LOG_FILE


IMAGE_CONFIG_FILE="simulate.yaml"

# NUM_OUTPUT_FILES=3

echo "Running galsim with $IMAGE_CONFIG_FILE..."
galsim "$IMAGE_CONFIG_FILE" variables.nfiles="$num_files"
