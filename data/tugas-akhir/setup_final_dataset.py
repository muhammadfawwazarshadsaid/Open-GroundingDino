import os
import json
import jsonlines

# --- CONFIG ---
train_src = "annotations/train_odvg.jsonl"     # file lama
train_dst = "annotations/train_odvg_clean.jsonl"  # output bersih
val_src   = "valid/_annotations.coco.json"        # file asli
val_dst   = "valid/_annotations.coco_clean.json"  # output bersih
label_map = "config/label_map.json"

# --- 1. Convert train (detection.instances -> annotations.text)
print("🔄 Converting train annotations...")
with jsonlines.open(train_src) as reader, jsonlines.open(train_dst, mode="w") as writer:
    for obj in reader:
        anns = []
        if "detection" in obj and "instances" in obj["detection"]:
            for inst in obj["detection"]["instances"]:
                anns.append({
                    "bbox": inst["bbox"],
                    "text": inst.get("category")  # ambil nama kelas
                })
        elif "annotations" in obj:  # kalau udah format baru
            anns = obj["annotations"]

        writer.write({
            "filename": obj["filename"],
            "height": obj.get("height", None),
            "width": obj.get("width", None),
            "annotations": anns
        })
print(f"✅ Train converted → {train_dst}")

# --- 2. Clean val (hapus bogus class 'tugas-akhir')
print("🔄 Cleaning val annotations...")
with open(val_src, "r") as f:
    val_data = json.load(f)

val_classes = [cat["name"] for cat in val_data["categories"]]
if "tugas-akhir" in val_classes:
    print("⚠️ Removing bogus class 'tugas-akhir'")
    val_classes = [c for c in val_classes if c != "tugas-akhir"]
    val_data["categories"] = [{"id": i+1, "name": c} for i, c in enumerate(val_classes)]

with open(val_dst, "w") as f:
    json.dump(val_data, f, indent=2)
print(f"✅ Val cleaned → {val_dst}")

# --- 3. Validate konsistensi dengan label_map
print("🔎 Validating consistency...")
with open(label_map, "r") as f:
    label_map_dict = json.load(f)
id2label = {int(k): v for k, v in label_map_dict.items()}
label_set = set(id2label.values())

train_classes = set()
with jsonlines.open(train_dst) as reader:
    for obj in reader:
        for ann in obj.get("annotations", []):
            if "text" in ann:
                train_classes.add(ann["text"])

val_classes = [cat["name"] for cat in val_data["categories"]]
val_set = set(val_classes)

print("Train classes found:", sorted(train_classes))
print("Val classes found:", sorted(val_set))

missing_in_train = label_set - train_classes
missing_in_val = label_set - val_set

if missing_in_train:
    print("❌ Missing in train:", missing_in_train)
if missing_in_val:
    print("❌ Missing in val:", missing_in_val)

if not missing_in_train and not missing_in_val:
    print("✅ Semua kelas konsisten antara TRAIN, VAL, dan LABEL MAP!")
