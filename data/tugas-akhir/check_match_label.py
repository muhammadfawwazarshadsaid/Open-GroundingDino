import json
import jsonlines

train_odvg = "annotations/train_odvg_clean.jsonl"
val_coco = "valid/_annotations.coco_clean.json"
label_map = "config/label_map.json"

with open(label_map, "r") as f:
    label_map_dict = json.load(f)

label_map_classes = set(label_map_dict.values())

odvg_classes = set()
with jsonlines.open(train_odvg) as reader:
    for obj in reader:
        odvg_classes.add(obj["caption"])

with open(val_coco, "r") as f:
    coco_data = json.load(f)

coco_classes = set([cat["name"] for cat in coco_data["categories"]])

print("=== Classes in label_map.json ===")
print(label_map_classes)

print("\n=== Classes in train_odvg_clean.jsonl ===")
print(odvg_classes)

print("\n=== Classes in _annotations.coco_clean.json ===")
print(coco_classes)

print("\n--- Checking consistency ---")
missing_in_label_map = (odvg_classes | coco_classes) - label_map_classes
missing_in_odvg = label_map_classes - odvg_classes
missing_in_coco = label_map_classes - coco_classes

if not missing_in_label_map and not missing_in_odvg and not missing_in_coco:
    print("✅ Semua konsisten!")
else:
    if missing_in_label_map:
        print("⚠️ Ada kelas di dataset (odvg/coco) tapi TIDAK ada di label_map:", missing_in_label_map)
    if missing_in_odvg:
        print("⚠️ Ada kelas di label_map tapi TIDAK muncul di ODVG train:", missing_in_odvg)
    if missing_in_coco:
        print("⚠️ Ada kelas di label_map tapi TIDAK muncul di COCO val:", missing_in_coco)
