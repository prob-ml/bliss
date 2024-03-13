#!/bin/bash

# Navigate to the directory containing your venv and the case_studies directory
cd ~/bliss 

# Activate the virtual environment
source venv/bin/activate

# Change directory to case_studies
cd case_studies

# Run your Python script
python3 get_redshift.py

# Optionally deactivate the virtual environment
deactivate

