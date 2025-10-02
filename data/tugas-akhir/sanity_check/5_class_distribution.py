import json
import jsonlines
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Paths
labelmap_path = "../config/label_map_fixed.json"
coco_path = "../valid/_annotations.coco_fixed.json"
odvg_path = "../annotations/train_odvg_fixed.jsonl"
output_excel = "../sanity_check/class_distribution_summary.xlsx"

print("=== Class Distribution Check (Augmentation Guidance) ===")

# Load label_map
with open(labelmap_path, "r") as f:
    label_map = json.load(f)
id2cat = {int(k): v for k, v in label_map.items()}
valid_classes = list(id2cat.values())
print(f"✅ Loaded {len(valid_classes)} classes from label_map_fixed.json")

# Count COCO
def distribution_coco(path):
    with open(path, "r") as f:
        data = json.load(f)
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    counter = Counter()
    for ann in data["annotations"]:
        name = id2name.get(ann["category_id"], None)
        if name:
            counter[name] += 1
    return counter

# Count ODVG
def distribution_odvg(path):
    counter = Counter()
    with jsonlines.open(path) as reader:
        for ex in reader:
            dets = ex.get("detection", {})
            if isinstance(dets, dict) and "instances" in dets:
                for det in dets["instances"]:
                    if det.get("category"):
                        counter[det["category"]] += 1
            elif isinstance(dets, list) and all(isinstance(d, dict) for d in dets):
                for det in dets:
                    if det.get("category"):
                        counter[det["category"]] += 1
            elif isinstance(dets, list) and all(isinstance(d, str) for d in dets):
                for det in dets:
                    counter[det] += 1
    return counter

train_dist = distribution_odvg(odvg_path)
val_dist = distribution_coco(coco_path)

# Summary
rows = []
for c in valid_classes:
    t = train_dist.get(c, 0)
    v = val_dist.get(c, 0)
    total = t + v
    rows.append([c, t, v, total])

df = pd.DataFrame(rows, columns=["Class", "Train (ODVG)", "Val (COCO)", "Total"])

print("\n=== Summary Table ===")
print(df.to_string(index=False))

# Baseline pakai median biar robust
baseline = df["Total"].median()
print(f"\n📌 Median jumlah anotasi per kelas = {baseline}")

# Assign warna + rekomendasi + rasio
reco = []
ratios = []
for total in df["Total"]:
    ratio = round(total / baseline, 2) if baseline > 0 else 0
    ratios.append(f"{ratio}x")
    if total < 0.5 * baseline:  
        reco.append("⚠️ Perlu banyak augmentasi")
    elif total < baseline:         
        reco.append("Perlu augmentasi")
    else:                          
        reco.append("✅ Cukup seimbang")

df["Ratio vs Median"] = ratios
df["Augment Recommendation"] = reco

# Save summary ke Excel
df.to_excel(output_excel, index=False)
print(f"\n💾 Summary table diexport ke: {output_excel}")

# Visualisasi
plt.figure(figsize=(12,6))
df_sorted = df.sort_values("Total", ascending=False)

colors = []
for total in df_sorted["Total"]:
    if total < 0.5 * baseline:
        colors.append("red")
    elif total < baseline:
        colors.append("orange")
    else:
        colors.append("seagreen")

plt.bar(df_sorted["Class"], df_sorted["Total"], color=colors)
plt.axhline(y=baseline, color="blue", linestyle="--", label=f"Median={baseline}")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Jumlah Anotasi")
plt.title("Distribusi Kelas & Rekomendasi Augmentasi\n(Merah=Perlu Banyak, Oranye=Sedikit, Hijau=Aman)")
plt.legend()
plt.tight_layout()
plt.show()

print("\n📊 Done: distribusi class + rekomendasi augmentasi sudah divisualisasikan.")
