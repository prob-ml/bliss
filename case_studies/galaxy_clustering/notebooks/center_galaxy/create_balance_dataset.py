#!/usr/bin/env python3
import torch
from pathlib import Path

# === Config ===
SRC_DIR = Path("/scratch/regier_root/regier0/hughwang/remerge_cg/file_data_1")
OUT_DIR = Path("/scratch/regier_root/regier0/hughwang/remerge_cg/file_data_balanced_35000")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SIZE = 19
TILE_SIZE = 256
MAX_PER_CLASS = 35000   # <-- use full number now

# === Global counters ===
true_count = 0
false_count = 0


def process_large_tile(pt_path: Path):
    global true_count, false_count
    try:
        data = torch.load(pt_path, map_location="cpu")
        img = data["images"]                      # [4,4864,4864]
        membership = data["tile_catalog"]["membership"]  # [19,19]

        base_name = pt_path.stem.replace("file_data_", "")
        saved = 0

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                label_bool = bool(membership[r, c])

                # Stop early if both quotas met
                if true_count >= MAX_PER_CLASS and false_count >= MAX_PER_CLASS:
                    return "🛑 Limit reached"

                if label_bool and true_count >= MAX_PER_CLASS:
                    continue
                if not label_bool and false_count >= MAX_PER_CLASS:
                    continue

                y0, y1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
                x0, x1 = c * TILE_SIZE, (c + 1) * TILE_SIZE

                # ✅ Copy the slice — avoid saving the whole 4864x4864 tensor
                tile_img = img[:, y0:y1, x0:x1].clone().contiguous()

                record = {
                    "image": tile_img,
                    "label": label_bool,
                    "tile_name": f"{base_name}_r{r:02d}_c{c:02d}"
                }

                out_path = OUT_DIR / f"{base_name}_r{r:02d}_c{c:02d}_label{label_bool}.pt"
                torch.save(record, out_path)
                saved += 1

                if label_bool:
                    true_count += 1
                else:
                    false_count += 1

                if true_count >= MAX_PER_CLASS and false_count >= MAX_PER_CLASS:
                    print(f"🟢 Reached limit ({true_count} True, {false_count} False). Stopping.")
                    return "🛑 Limit reached"

        return f"✅ {pt_path.name}: saved {saved} tiles (True={true_count}, False={false_count})"

    except Exception as e:
        return f"❌ {pt_path.name}: {e}"


# === Sequential processing (stop when limits hit) ===
pt_files = list(SRC_DIR.glob("*.pt"))
print(f"Found {len(pt_files)} large tiles")

for f in pt_files:
    result = process_large_tile(f)
    print(result)
    if "Limit reached" in result:
        break

print(f"✅ Done! Final counts: True={true_count}, False={false_count}")
print(f"Tiles saved to: {OUT_DIR}")
