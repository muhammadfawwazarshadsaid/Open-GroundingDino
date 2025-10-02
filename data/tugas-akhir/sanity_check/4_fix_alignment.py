import json
import jsonlines

labelmap_path = "../config/label_map_fixed.json"
coco_src = "../valid/_annotations.coco.json"
coco_out = "../valid/_annotations.coco_fixed.json"
odvg_src = "../annotations/train_odvg.jsonl"
odvg_out = "../annotations/train_odvg_fixed.jsonl"

print("=== Fix Alignment: COCO + ODVG ===")

#  Load label_map
with open(labelmap_path, "r") as f:
    label_map = json.load(f)
id2cat = {int(k): v for k, v in label_map.items()}
cat2id = {v: int(k) for k, v in label_map.items()}
valid_classes = set(cat2id.keys())
print(f"✅ Loaded {len(id2cat)} classes from label_map_fixed.json")

# Step 1: Fix COCO
with open(coco_src, "r") as f:
    coco = json.load(f)

print(f"\n📂 COCO before fix: {len(coco['categories'])} categories, {len(coco['annotations'])} annotations")

new_categories, old2new = [], {}
for cat in coco["categories"]:
    name = cat["name"]
    if name in cat2id:
        new_id = cat2id[name]
        old2new[cat["id"]] = new_id
        new_categories.append({"id": new_id, "name": name})
    else:
        print(f"⚠️ Removing unknown COCO category: {name}")

coco["categories"] = new_categories

new_annotations = []
for ann in coco["annotations"]:
    old_id = ann["category_id"]
    if old_id in old2new:
        ann["category_id"] = old2new[old_id]
        new_annotations.append(ann)
    else:
        print(f"⚠️ Removing COCO annotation with unknown category_id {old_id}")

coco["annotations"] = new_annotations

with open(coco_out, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✅ COCO aligned → {coco_out}")
print(f"📂 COCO after fix: {len(coco['categories'])} categories, {len(coco['annotations'])} annotations")

# Step 2: Fix ODVG
print(f"\n📂 ODVG before fix: {sum(1 for _ in open(odvg_src))} entries")
cleaned = []
with jsonlines.open(odvg_src) as reader:
    for ex in reader:
        if "detection" in ex:
            dets = ex["detection"]

            # case 1: dict with instances
            if isinstance(dets, dict) and "instances" in dets:
                new_instances = [d for d in dets["instances"] if d.get("category") in valid_classes]
                if new_instances:
                    ex["detection"]["instances"] = new_instances
                    cleaned.append(ex)

            # case 2: list of dicts
            elif isinstance(dets, list) and all(isinstance(d, dict) for d in dets):
                new_dets = [d for d in dets if d.get("category") in valid_classes]
                if new_dets:
                    ex["detection"] = new_dets
                    cleaned.append(ex)

            # case 3: list of strings
            elif isinstance(dets, list) and all(isinstance(d, str) for d in dets):
                new_dets = [d for d in dets if d in valid_classes]
                if new_dets:
                    ex["detection"] = new_dets
                    cleaned.append(ex)

with jsonlines.open(odvg_out, "w") as writer:
    writer.write_all(cleaned)

print(f"✅ ODVG aligned → {odvg_out}")
print(f"📂 ODVG after fix: {len(cleaned)} entries")

print("\n🚀 Alignment done: COCO + ODVG fully match label_map_fixed.json")
