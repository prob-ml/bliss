import pandas as pd
from astropy.table import Table
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Paths ===
merged_dir = Path("/scratch/regier_root/regier0/hughwang/remerge_cg/catalogs")
clusters_path = Path("/scratch/regier_root/regier0/hughwang/desdr-server.ncsa.illinois.edu/despublic/y6a2_dnf_wazp_v5.0.12.6801_clusters.fits")
out_dir = Path("/scratch/regier_root/regier0/hughwang/remerge_cg/catalogs_filtered")
out_dir.mkdir(parents=True, exist_ok=True)

# === Load clusters and apply ZPHOT/NGALS filter ===
clusters_df = Table.read(clusters_path, hdu=1).to_pandas()
filtered_clusters = clusters_df.query("ZPHOT < 1 and NGALS > 20")
keep_ids = set(filtered_clusters["ID_CG"].values)   # ID_CG in cluster file
print(f"Clusters passing filter: {len(keep_ids)}")

# === Function to filter each merged catalog ===
def filter_catalog(cat_file: Path):
    try:
        out_path = out_dir / cat_file.name
        if out_path.exists():
            return f"⏩ Skipped (exists): {out_path}"

        # Load merged catalog
        df = pd.read_csv(cat_file, sep=" ", header=None)

        # Assume last 2 columns are COADD_OBJECT_ID and FLAG_CG
        col_count = df.shape[1]
        col_names = list(range(col_count))
        col_names[-2:] = ["COADD_OBJECT_ID", "FLAG_CG"]
        df.columns = col_names

        # Filter by COADD_OBJECT_ID (== ID_CG)
        df_filtered = df[df["COADD_OBJECT_ID"].isin(keep_ids)]

        if len(df_filtered) > 0:
            df_filtered.to_csv(out_path, sep=" ", index=False, header=False)
            return f"✅ Saved: {out_path} ({len(df_filtered)} rows)"
        else:
            return f"⚠️ Empty after filter: {cat_file.name}"

    except Exception as e:
        return f"❌ Error processing {cat_file}: {e}"

# === Run in parallel ===
cat_files = list(merged_dir.glob("*.cat"))
print(f"Found {len(cat_files)} merged catalogs to filter")

with ProcessPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(filter_catalog, cat_file): cat_file for cat_file in cat_files}
    for future in as_completed(futures):
        print(future.result())
