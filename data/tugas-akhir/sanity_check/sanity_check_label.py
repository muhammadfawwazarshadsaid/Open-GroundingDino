import json
import jsonlines

labelmap_path = "../config/label_map_fixed.json"
coco_fixed = "../valid/_annotations.coco_fixed.json"
odvg_fixed = "../annotations/train_odvg_fixed.jsonl"

print("=== Sanity Check: Label Alignment ===")

# Load label_map
with open(labelmap_path, "r") as f:
    label_map = json.load(f)

id2cat = {int(k): v for k, v in label_map.items()}
max_id = max(id2cat.keys())
print(f"✅ Loaded {len(id2cat)} classes from label_map_fixed.json")

# Print detail 0-n
print("\n📋 Daftar Class ID:")
for i in range(len(id2cat)):
    print(f"  {i}: {id2cat[i]}")

# --- Check COCO ---
with open(coco_fixed, "r") as f:
    coco = json.load(f)

ann_ids = [a["category_id"] for a in coco["annotations"]]
invalid_coco = [a for a in coco["annotations"] if a["category_id"] not in id2cat]

print(f"\n📂 COCO annotations: {len(ann_ids)} total")
print(f"   Unique IDs: {sorted(set(ann_ids))}")
if invalid_coco:
    print(f"❌ Found {len(invalid_coco)} invalid COCO labels")
else:
    print("✅ All COCO labels valid")

# --- Check ODVG ---
invalid_odvg = []
all_ids_odvg = []
with jsonlines.open(odvg_fixed) as reader:
    for i, ex in enumerate(reader):
        dets = ex.get("detection", {})
        if isinstance(dets, dict) and "instances" in dets:
            for d in dets["instances"]:
                lb = d.get("label")
                if lb is not None:
                    all_ids_odvg.append(lb)
                    if lb not in id2cat:
                        invalid_odvg.append((i, d))
        elif isinstance(dets, list):
            for d in dets:
                lb = d.get("label")
                if lb is not None:
                    all_ids_odvg.append(lb)
                    if lb not in id2cat:
                        invalid_odvg.append((i, d))

print(f"\n📂 ODVG detections: {len(all_ids_odvg)} total labels")
print(f"   Unique IDs: {sorted(set(all_ids_odvg))}")
if invalid_odvg:
    print(f"❌ Found {len(invalid_odvg)} invalid ODVG labels")
    print("  Contoh:", invalid_odvg[:5])
else:
    print("✅ All ODVG labels valid")

print("\n🚀 Sanity check selesai, semua label harus match dengan label_map.")
