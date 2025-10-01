import json
import os
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jsonlines

# --- CONFIG ---
val_src   = "valid/_annotations.coco.json"
val_clean = "valid/_annotations.coco_clean.json"

train_odvg_src  = "annotations/train_odvg.jsonl"
train_odvg_clean = "annotations/train_odvg_clean.jsonl"

label_map = "config/label_map.json"

# --- Load label_map ---
with open(label_map, "r") as f:
    label_map_dict = json.load(f)
valid_classes = set(label_map_dict.values())
print(f"✅ Loaded {len(valid_classes)} valid classes from label_map.json")

# --- Step 1. Clean val COCO ---
def clean_coco(src, dst, valid_classes):
    with open(src, "r") as f:
        data = json.load(f)

    before = len(data["categories"])
    cats_before = [c["name"] for c in data["categories"]]

    # filter kategori sesuai label_map
    data["categories"] = [c for c in data["categories"] if c["name"] in valid_classes]
    after = len(data["categories"])
    cats_after = [c["name"] for c in data["categories"]]

    valid_ids = {c["id"] for c in data["categories"]}
    data["annotations"] = [a for a in data["annotations"] if a["category_id"] in valid_ids]

    with open(dst, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n🧹 Cleaned {os.path.basename(src)} → {os.path.basename(dst)}")
    print(f"   Categories: {before} → {after}")
    print(f"   Removed: {set(cats_before) - set(cats_after)}")
    print(f"   Annotations left: {len(data['annotations'])}")

    return dst

val_clean = clean_coco(val_src, val_clean, valid_classes)

# --- Step 2a. Inspect ODVG keys ---
def inspect_odvg_keys(src, sample=5):
    keys = set()
    with jsonlines.open(src) as reader:
        for i, ex in enumerate(reader):
            keys.update(ex.keys())
            if i < sample:
                print(f"🔎 Sample {i+1} keys: {list(ex.keys())}")
    print(f"📑 Unique keys in ODVG: {keys}\n")
    return keys

inspect_odvg_keys(train_odvg_src)

# --- Step 2b. Clean train ODVG ---
def clean_odvg(src, dst, valid_classes):
    with jsonlines.open(src) as reader:
        items = list(reader)

    before = len(items)
    cleaned = []

    for ex in items:
        if "detection" in ex:
            dets = ex["detection"]

            # case 1: list of dicts
            if isinstance(dets, list) and all(isinstance(d, dict) for d in dets):
                new_dets = [d for d in dets if d.get("category") in valid_classes]
                if new_dets:
                    ex["detection"] = new_dets
                    ex["caption"] = new_dets[0]["category"]
                    cleaned.append(ex)

            # case 2: list of strings
            elif isinstance(dets, list) and all(isinstance(d, str) for d in dets):
                new_dets = [d for d in dets if d in valid_classes]
                if new_dets:
                    ex["detection"] = new_dets
                    ex["caption"] = new_dets[0]
                    cleaned.append(ex)

            # case 3: dict with "instances"
            elif isinstance(dets, dict) and "instances" in dets:
                new_dets = [d for d in dets["instances"] if d.get("category") in valid_classes]
                if new_dets:
                    ex["detection"]["instances"] = new_dets
                    ex["caption"] = new_dets[0]["category"]
                    cleaned.append(ex)

    after = len(cleaned)

    with jsonlines.open(dst, "w") as writer:
        writer.write_all(cleaned)

    print(f"\n🧹 Cleaned {os.path.basename(src)} → {os.path.basename(dst)}")
    print(f"   Annotations: {before} → {after}")
    return dst

train_odvg_clean = clean_odvg(train_odvg_src, train_odvg_clean, valid_classes)

# --- Step 3. Distribusi kelas ---
def distribution_coco(path):
    with open(path, "r") as f:
        data = json.load(f)
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    counter = Counter()
    for ann in data["annotations"]:
        counter[id2name.get(ann["category_id"], "UNKNOWN")] += 1
    return counter

def distribution_odvg(path):
    counter = Counter()
    with jsonlines.open(path) as reader:
        for ex in reader:
            dets = ex.get("detection", [])

            # case 1: list of dicts
            if isinstance(dets, list) and all(isinstance(d, dict) for d in dets):
                for det in dets:
                    if det.get("category"):
                        counter[det["category"]] += 1

            # case 2: list of strings
            elif isinstance(dets, list) and all(isinstance(d, str) for d in dets):
                for det in dets:
                    counter[det] += 1

            # case 3: dict dengan instances
            elif isinstance(dets, dict) and "instances" in dets:
                for det in dets["instances"]:
                    if det.get("category"):
                        counter[det["category"]] += 1

    return counter



train_dist = distribution_odvg(train_odvg_clean)
val_dist   = distribution_coco(val_clean)

# --- Step 4. Summary table ---
classes = sorted(set(train_dist.keys()) | set(val_dist.keys()))
rows = []
for c in classes:
    t = train_dist.get(c, 0)
    v = val_dist.get(c, 0)
    total = t + v
    t_pct = (t/total*100) if total > 0 else 0
    v_pct = (v/total*100) if total > 0 else 0
    rows.append([c, t, v, f"{t_pct:.1f}%", f"{v_pct:.1f}%"])

df = pd.DataFrame(rows, columns=["Class", "Train Count (ODVG)", "Val Count (COCO)", "Train %", "Val %"])
print("\n=== Summary Table ===")
print(df.to_string(index=False))
