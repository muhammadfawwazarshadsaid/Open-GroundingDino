import json
import jsonlines
import os

# Paths
labelmap_path = "../config/label_map.json"
coco_src = "../valid/_annotations.coco.json"
coco_out = "../valid/_annotations.coco_fixed.json"
odvg_src = "../annotations/train_odvg.jsonl"
odvg_out = "../annotations/train_odvg_fixed.jsonl"

print("\n=== Fixing Dataformat (Full) ===")

# Step 1. Fix Label Map
def fix_labelmap(path):
    with open(path, "r") as f:
        data = json.load(f)

    items = list(data.items())
    keys = [int(k) for k, _ in items]

    if min(keys) != 0 or sorted(keys) != list(range(len(keys))):
        print("❌ Label map tidak konsisten → auto-fix ...")
        fixed = {str(i): v for i, (_, v) in enumerate(items)}
        out_path = path.replace(".json", "_fixed.json")
        with open(out_path, "w") as f:
            json.dump(fixed, f, indent=2)
        print(f"✅ label_map fixed → {os.path.basename(out_path)}")
        return out_path, fixed
    else:
        print("✅ Label map sudah valid")
        return path, data

labelmap_fixed, labelmap_dict = fix_labelmap(labelmap_path)
id2cat = {int(k): v for k, v in labelmap_dict.items()}
cat2id = {v: k for k, v in id2cat.items()}
num_classes = len(id2cat)

# Step 2. Fix COCO
def fix_coco(src, dst, cat2id):
    with open(src, "r") as f:
        data = json.load(f)

    # Fix categories
    name2id = {}
    new_categories = []
    for i, (cid, cname) in enumerate([(c["id"], c["name"]) for c in data["categories"]]):
        if cname in cat2id:
            new_id = cat2id[cname]
            new_categories.append({"id": new_id, "name": cname})
            name2id[cid] = new_id

    # Fix annotations
    new_annotations = []
    for ann in data["annotations"]:
        old_id = ann["category_id"]
        if old_id in name2id:
            ann["category_id"] = name2id[old_id]
            new_annotations.append(ann)

    data["categories"] = new_categories
    data["annotations"] = new_annotations

    with open(dst, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ COCO fixed → {os.path.basename(dst)}")
    print(f"   Categories: {len(new_categories)}, Annotations: {len(new_annotations)}")
    return dst

coco_fixed = fix_coco(coco_src, coco_out, cat2id)

# Step 3. Fix ODVG
def fix_odvg(src, dst, cat2id):
    fixed = []
    with jsonlines.open(src) as reader:
        for ex in reader:
            if "detection" in ex:
                dets = ex["detection"]

                # case 1: dict with "instances"
                if isinstance(dets, dict) and "instances" in dets:
                    for det in dets["instances"]:
                        if "category" in det and det["category"] in cat2id:
                            det["label"] = cat2id[det["category"]]
                    fixed.append(ex)

                # case 2: list of dicts
                elif isinstance(dets, list) and all(isinstance(d, dict) for d in dets):
                    for det in dets:
                        if "category" in det and det["category"] in cat2id:
                            det["label"] = cat2id[det["category"]]
                    fixed.append(ex)

                # case 3: list of strings
                elif isinstance(dets, list) and all(isinstance(d, str) for d in dets):
                    new_dets = []
                    for d in dets:
                        if d in cat2id:
                            new_dets.append({"category": d, "label": cat2id[d]})
                    if new_dets:
                        ex["detection"] = {"instances": new_dets}
                        fixed.append(ex)

    with jsonlines.open(dst, "w") as writer:
        writer.write_all(fixed)

    print(f"✅ ODVG fixed → {os.path.basename(dst)}")
    print(f"   Total entries: {len(fixed)}")
    return dst

odvg_fixed = fix_odvg(odvg_src, odvg_out, cat2id)

print("\n=== Selesai ✅ ===")
print(f"Gunakan file:\n- {labelmap_fixed}\n- {coco_fixed}\n- {odvg_fixed}")
