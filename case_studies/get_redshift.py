from astropy.io import fits
f = fits.open('/home/../data/scratch/specObj-dr17.fits')
len(f[1].data)
vars(f[1].data)
print(f[1].data[15]['Z']) #redshift for 15th object in this file
print(f[1].data[15]['SOURCETYPE']) #tells us if 15th object is galaxy, star, or quasar
print(f[1].data[15]['ZWARNING']) #tells us if there is a warning with this redshift; if yes, we'd exclude it.
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
condition = (data['SPECPRIMARY'] > 0) & ((data['ZWARNING'] == 0) | (data['ZWARNING'] == 16))

# Apply the condition to filter the 'Z' values
filtered_z_values = data['Z'][condition]
filtered_ID= data['SPECOBJID'][condition]
from astroquery.sdss import SDSS
import csv

# Assuming 'data' contains your initial data with 'SPECOBJID' column
filtered_ID = data['SPECOBJID'][condition]  # Assuming you have defined 'condition' somewhere

# Open a CSV file to write the data
with open('/home/../data/scratch/photometric_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['objID', 'u_band', 'g_band', 'r_band', 'i_band', 'z_band'])
    
    # Initialize a counter for tracking the number of rows written
    row_count = 0
    
    # Loop through each SPECOBJID and query photometric data
    for objid in filtered_ID:
        # Construct an SQL query to fetch the desired photometric data for the objid
        sql_query = f"""
        SELECT
            p.objID, p.u, p.g, p.r, p.i, p.z
        FROM
            PhotoObj AS p
            JOIN SpecObj AS s ON p.objID = s.bestObjID
        WHERE
            s.specObjID = {int(objid)}
        """
        
        try:
            photo_data = SDSS.query_sql(sql_query)
            if photo_data is not None and len(photo_data) > 0:
                # Write the data row to the CSV file
                writer.writerow([
                    photo_data['objID'][0],
                    photo_data['u'][0],
                    photo_data['g'][0],
                    photo_data['r'][0],
                    photo_data['i'][0],
                    photo_data['z'][0]
                ])
            else:
                # Write a row with None values if no data is available
                writer.writerow([objid, None, None, None, None, None])
        except Exception as e:
            print(f"Failed to query or process data for objid {objid}: {e}")
            # Write a row with None values in case of failure
            writer.writerow([objid, None, None, None, None, None])
        
        row_count += 1
        # Print a message every 1000 rows
        if row_count % 1000 == 0:
            print(f"Written {row_count} rows to the CSV file.")
