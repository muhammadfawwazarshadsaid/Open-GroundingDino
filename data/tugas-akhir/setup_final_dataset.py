import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- CONFIG ---
train_src = "train/_annotations.coco.json"
val_src   = "valid/_annotations.coco.json"

train_dst = "train/_annotations.coco_clean.json"
val_dst   = "valid/_annotations.coco_clean.json"

label_map = "config/label_map.json"

# --- 1. Load label_map
with open(label_map, "r") as f:
    label_map_dict = json.load(f)
valid_classes = set(label_map_dict.values())
print(f"✅ Loaded {len(valid_classes)} valid classes from label_map.json")

def clean_coco(src, dst, valid_classes):
    with open(src, "r") as f:
        data = json.load(f)

    before = len(data["categories"])
    cats_before = [c["name"] for c in data["categories"]]

    # filter kategori bogus
    data["categories"] = [c for c in data["categories"] if c["name"] in valid_classes]
    after = len(data["categories"])
    cats_after = [c["name"] for c in data["categories"]]

    # filter annotations yang kelasnya gak ada di label_map
    valid_ids = {c["id"] for c in data["categories"]}
    data["annotations"] = [a for a in data["annotations"] if a["category_id"] in valid_ids]

    with open(dst, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n🧹 Cleaned {os.path.basename(src)} → {os.path.basename(dst)}")
    print(f"   Categories: {before} → {after}")
    print(f"   Removed: {set(cats_before) - set(cats_after)}")
    print(f"   Annotations left: {len(data['annotations'])}")

    return dst

# --- Clean both
train_clean = clean_coco(train_src, train_dst, valid_classes)
val_clean   = clean_coco(val_src, val_dst, valid_classes)

print("\n✅ All done! Train & Val COCO cleaned")

# --- 2. Distribusi bbox
def load_distribution(path):
    with open(path, "r") as f:
        data = json.load(f)
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    counter = Counter()
    for ann in data["annotations"]:
        cid = ann["category_id"]
        counter[id2name.get(cid, "UNKNOWN")] += 1
    return counter

train_dist = load_distribution(train_clean)
val_dist = load_distribution(val_clean)

# --- 3. Tabel ringkasan
classes = sorted(set(train_dist.keys()) | set(val_dist.keys()))
rows = []
for c in classes:
    t = train_dist.get(c, 0)
    v = val_dist.get(c, 0)
    total = t + v
    t_pct = (t/total*100) if total > 0 else 0
    v_pct = (v/total*100) if total > 0 else 0
    rows.append([c, t, v, f"{t_pct:.1f}%", f"{v_pct:.1f}%"])

df = pd.DataFrame(rows, columns=["Class", "Train Count", "Val Count", "Train %", "Val %"])
print("\n=== Summary Table ===")
print(df.to_string(index=False))

# --- 4. Plot side-by-side absolute
def plot_side_by_side(train_dist, val_dist, title):
    train_vals = [train_dist.get(c, 0) for c in classes]
    val_vals = [val_dist.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(14,6))
    plt.bar(x - width/2, train_vals, width, label="Train", color="steelblue")
    plt.bar(x + width/2, val_vals, width, label="Val", color="orange")

    plt.xticks(x, classes, rotation=75, ha="right")
    plt.title(title)
    plt.ylabel("Jumlah bbox")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_side_by_side(train_dist, val_dist, "Distribusi Anotasi per Kelas (Train vs Val)")

# --- 5. Plot persentase
def plot_percentage(train_dist, val_dist, title):
    percentages = []
    for c in classes:
        t = train_dist.get(c, 0)
        v = val_dist.get(c, 0)
        total = t + v
        if total == 0:
            percentages.append((0,0))
        else:
            percentages.append((t/total*100, v/total*100))

    x = np.arange(len(classes))
    width = 0.35

    train_pct = [p[0] for p in percentages]
    val_pct = [p[1] for p in percentages]

    plt.figure(figsize=(14,6))
    plt.bar(x - width/2, train_pct, width, label="Train %", color="steelblue")
    plt.bar(x + width/2, val_pct, width, label="Val %", color="orange")

    plt.xticks(x, classes, rotation=75, ha="right")
    plt.title(title)
    plt.ylabel("Persentase (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_percentage(train_dist, val_dist, "Proporsi Persentase Train vs Val per Kelas")
