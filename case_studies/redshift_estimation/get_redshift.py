from astropy.io import fits

f = fits.open("/home/../data/scratch/specObj-dr17.fits")
import numpy as np

data = f[1].data

# # Extract the 'TARGETTYPE' column
# target_types = data['TARGETTYPE']

# # Find unique values in the 'TARGETTYPE' column
# unique_target_types = np.unique(target_types)

# # Print the unique values
# print('Unique TARGETTYPE values:')
# for target_type in unique_target_types:
#     print(target_type)
# Apply conditions:
# 'SPECPRIMARY' > 0
# 'ZWARNING' == 0 or 'ZWARNING' == 16
condition = (data["SPECPRIMARY"] > 0) & ((data["ZWARNING"] == 0) | (data["ZWARNING"] == 16))

# Apply the condition to filter the 'Z' values
filtered_z_values = data["Z"][condition]
filtered_ID = data["SPECOBJID"][condition]


import csv

from astroquery.sdss import SDSS

# Assuming 'data' contains your initial data with 'SPECOBJID' column
filtered_ID = data["SPECOBJID"][condition]  # Assuming you have defined 'condition' somewhere
total_ids = len(filtered_ID)

# Specify the output CSV file path
output_file_path = "/data/scratch/declan/photometric_data.csv"


# Function to write a batch of data to the CSV file
def write_batch_to_csv(data_batch, header=False):
    with open(output_file_path, mode="a", newline="") as file:  # 'a' mode to append data
        writer = csv.writer(file)
        if header:
            writer.writerow(
                [
                    "objID",
                    "u_band",
                    "g_band",
                    "r_band",
                    "i_band",
                    "z_band",
                    "redshift",
                    "source_type",
                ]
            )
        writer.writerows(data_batch)


# Write the header row
write_batch_to_csv([], header=True)

for start_index in range(0, total_ids, 100):
    end_index = min(start_index + 100, total_ids)
    objid_list = ",".join([str(int(objid)) for objid in filtered_ID[start_index:end_index]])

    # Construct the SQL query for the current batch
    sql_query = f"""
    SELECT
        p.objID, p.u, p.g, p.r, p.i, p.z, s.z AS redshift, s.class AS source_type
    FROM
        PhotoObj AS p
        JOIN SpecObj AS s ON p.objID = s.bestObjID
    WHERE
        s.specObjID IN ({objid_list})
    """

    try:
        photo_data = SDSS.query_sql(sql_query)
        if photo_data is not None and len(photo_data) > 0:
            # Prepare data batch for CSV writing
            data_batch = []
            for row in photo_data:
                objID = row["objID"]
                u = row["u"]
                g = row["g"]
                r = row["r"]
                i = row["i"]
                z = row["z"]
                redshift = row["redshift"]
                source_type = row["source_type"]

                if (u < 0) or (g < 0) or (r < 0) or (i < 0) or (z < 0) or (redshift < 0):
                    continue
                else:
                    data_row = [objID, u, g, r, i, z, redshift, source_type]
                    data_batch.append(data_row)

            # Write the current batch of data to the CSV file
            write_batch_to_csv(data_batch)
    except Exception as e:
        print(f"Failed to query or process data for batch starting at index {start_index}: {e}")

# Inform the user that the process is complete
print("Data querying and writing complete.")
