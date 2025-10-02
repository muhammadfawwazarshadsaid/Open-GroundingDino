import shutil
import json
import os

# Paths sumber hasil fix
labelmap_fixed = "../config/label_map_fixed.json"
coco_fixed = "../valid/_annotations.coco_fixed.json"
odvg_fixed = "../annotations/train_odvg_fixed.jsonl"

# Paths tujuan final
labelmap_final = "../config/label_map_final.json"
coco_final = "../valid/_annotations.coco_final.json"
odvg_final = "../annotations/train_odvg_final.jsonl"
datasets_out = "../config/datasets_od.json"

print("=== Setup Final Dataset ===")

# 1. Copy file hasil fix ke final
shutil.copyfile(labelmap_fixed, labelmap_final)
shutil.copyfile(coco_fixed, coco_final)
shutil.copyfile(odvg_fixed, odvg_final)

print(f"✅ Copied {labelmap_fixed} → {labelmap_final}")
print(f"✅ Copied {coco_fixed} → {coco_final}")
print(f"✅ Copied {odvg_fixed} → {odvg_final}")

# 2. Generate datasets_od.json
datasets_cfg = {
    "train": [
        {
            "root": "data/tugas-akhir/train/",
            "anno": "data/tugas-akhir/annotations/train_odvg_final.jsonl",
            "label_map": "data/tugas-akhir/config/label_map_final.json",
            "dataset_mode": "odvg"
        }
    ],
    "val": [
        {
            "root": "data/tugas-akhir/valid/",
            "anno": "data/tugas-akhir/valid/_annotations.coco_final.json",
            "label_map": "data/tugas-akhir/config/label_map_final.json",
            "dataset_mode": "coco"
        }
    ]
}

with open(datasets_out, "w") as f:
    json.dump(datasets_cfg, f, indent=2)

print(f"✅ Generated {datasets_out}")
print("\n🚀 Final dataset setup complete! Pakai datasets_od.json untuk training.")
