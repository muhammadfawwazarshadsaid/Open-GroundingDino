import json
from collections import Counter

# path ke file
jsonl_path = "annotations/train_odvg.jsonl"
label_map_path = "config/label_map.json"

# load label_map
with open(label_map_path, "r") as f:
    label_map = json.load(f)

# counter buat label
counter = Counter()

# baca JSONL (tiap baris = 1 gambar)
with open(jsonl_path, "r") as f:
    for line in f:
        data = json.loads(line)
        for inst in data["detection"]["instances"]:
            counter[inst["label"]] += 1

# tampilkan hasil
print("Distribusi kelas:")
for k, v in sorted(counter.items()):
    print(f"{k} - {label_map[str(k)]}: {v} instances")

print("\nTotal instances:", sum(counter.values()))
print("Total classes muncul:", len(counter))
