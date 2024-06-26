import numpy as np
from astropy.table import Table

col_names = [
    "IDENT",
    "RA",
    "DEC",
    "MAG",
    "BAND",
    "WEIGHT",
    "GAL_FILENAME",
    "PSF_FILENAME",
    "GAL_HDU",
    "PSF_HDU",
    "PIXEL_SCALE",
    "NOISE_MEAN",
    "NOISE_VARIANCE",
    "NOISE_FILENAME",
    "stamp_flux",
    "LENSED",
    "X",
    "Y",
    "N1",
    "HALF_RAD",
    "FLUX1",
    "N2",
    "SCALE_RAD",
    "FLUX2",
    "Q",
    "BETA",
    "THETA",
    "X_LENSE",
    "Y_LENSE",
    "E1",
    "E2",
]

col_dtypes = [
    "i4",
    "f8",
    "f8",
    "f8",
    "S5",
    "f8",
    "S32",
    "S36",
    "i4",
    "i4",
    "f8",
    "f8",
    "f8",
    "S36",
    "f8",
    "S5",
    "i4",
    "i4",
    "f8",
    "f8",
    "f8",
    "f8",
    "f8",
    "f8",
    "f8",
    "i4",
    "f8",
    "f8",
    "f8",
    "f8",
    "f8",
]

# Read the data from the text file
data = []
with open("data/catalog.txt", "r") as file:
    for line in file:
        # Remove any quotes from string fields and split the line by comma
        values = line.replace("'", "").split(", ")
        # Check if image is lensed and extract lensing parameters
        if values[15] == "True":
            row = (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                values[8],
                values[9],
                values[10],
                values[11],
                values[12],
                values[13],
                values[14],
                values[15],
                values[16],
                values[17],
                values[18],
                values[19],
                values[20],
                values[21],
                values[22],
                values[23],
                values[24],
                values[25],
                values[26],
                values[27],
                values[28],
                values[29],
                values[30],
            )
        else:
            row = (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                values[8],
                values[9],
                values[10],
                values[11],
                values[12],
                values[13],
                values[14],
                values[15],
                values[16],
                values[17],
                values[18],
                values[19],
                values[20],
                values[21],
                values[22],
                values[23],
                values[24],
                values[25],
                0,
                0,
                0,
                0,
                0,
            )
        data.append(row)

# Convert the list to a structured numpy array
structured_data = np.array(data, dtype={"names": col_names, "formats": col_dtypes})

# Create an Astropy Table from the structured numpy array
table = Table(structured_data)

# Save the table to a FITS file
table.write("data/catalog.fits", format="fits", overwrite=True)
