import shutil
import json
import os
import jsonlines

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

# 3. Alignment Check
print("\n=== Alignment Check ===")
with open(labelmap_final, "r") as f:
    label_map = json.load(f)
id2cat = {int(k): v for k, v in label_map.items()}
cat2id = {v: int(k) for k, v in label_map.items()}

# COCO
with open(coco_final, "r") as f:
    coco = json.load(f)
coco_classes = {c["name"] for c in coco["categories"]}

# ODVG
odvg_classes = set()
with jsonlines.open(odvg_final) as reader:
    for ex in reader:
        if "detection" in ex:
            dets = ex["detection"]
            if isinstance(dets, dict) and "instances" in dets:
                for det in dets["instances"]:
                    if "category" in det:
                        odvg_classes.add(det["category"])
            elif isinstance(dets, list) and all(isinstance(d, dict) for d in dets):
                for det in dets:
                    if "category" in det:
                        odvg_classes.add(det["category"])

print(f"✅ Label map classes: {len(id2cat)}")
print(f"✅ COCO classes     : {len(coco_classes)}")
print(f"✅ ODVG classes     : {len(odvg_classes)}")

mismatch_labelmap_coco = coco_classes - set(cat2id.keys())
mismatch_labelmap_odvg = odvg_classes - set(cat2id.keys())

if not mismatch_labelmap_coco and not mismatch_labelmap_odvg:
    print("\n🎉 Semua sudah sinkron, aman buat training 🚀")
else:
    if mismatch_labelmap_coco:
        print(f"⚠️ COCO extra/mismatch classes: {mismatch_labelmap_coco}")
    if mismatch_labelmap_odvg:
        print(f"⚠️ ODVG extra/mismatch classes: {mismatch_labelmap_odvg}")
    print("\n⚠️ Masih ada mismatch, cek lagi sebelum training!")

print("\n🚀 Final dataset setup + alignment check complete!")
