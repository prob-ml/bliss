import numpy as np
from astropy.table import Table

t = {
    "IDENT": "i4",
    "RA": "f8",
    "DEC": "f8",
    "MAG": "f8",
    "BAND": "S5",
    "WEIGHT": "f8",
    "GAL_FILENAME": "S32",
    "PSF_FILENAME": "S36",
    "GAL_HDU": "i4",
    "PSF_HDU": "i4",
    "PIXEL_SCALE": "f8",
    "NOISE_MEAN": "f8",
    "NOISE_VARIANCE": "f8",
    "NOISE_FILENAME": "S36",
    "stamp_flux": "f8",
    "LENSED": "S5",
    "X": "i4",
    "Y": "i4",
    "N1": "f8",
    "HALF_RAD": "f8",
    "FLUX1": "f8",
    "N2": "f8",
    "SCALE_RAD": "f8",
    "FLUX2": "f8",
    "Q": "f8",
    "BETA": "i4",
    "THETA": "f8",
    "X_LENSE": "f8",
    "Y_LENSE": "f8",
    "E1": "f8",
    "E2": "f8",
}

# Read the data from the text file
data = []
with open("data/catalog.txt", "r", encoding="utf-8") as file:
    for line in file:
        # Remove any quotes from string fields and split the line by comma
        values = line.replace("'", "").split(", ")
        # Check if image is lensed and extract lensing parameters
        if values[15] == "True":
            row = values[:31]
        else:
            row = values[:26] + [0, 0, 0, 0, 0]
        data.append(tuple(row))

# Convert the list to a structured numpy array
structured_data = np.array(data, dtype={"names": list(t.keys()), "formats": list(t.values())})

# Create an Astropy Table from the structured numpy array
table = Table(structured_data)

# Save the table to a FITS file
table.write("data/catalog.fits", format="fits", overwrite=True)
