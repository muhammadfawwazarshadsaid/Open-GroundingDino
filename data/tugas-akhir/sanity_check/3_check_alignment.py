import json
import jsonlines

odvg_path = "../annotations/train_odvg.jsonl"
coco_path = "../valid/_annotations.coco.json"
labelmap_path = "../config/label_map_fixed.json"

print("=== Alignment Check (with IDs) ===")

# Load label_map
with open(labelmap_path, "r") as f:
    label_map = json.load(f)
id2cat = {int(k): v for k, v in label_map.items()}
cat2id = {v: k for k, v in id2cat.items()}
print(f"✅ Loaded {len(id2cat)} classes from label_map")

# Load COCO
with open(coco_path, "r") as f:
    coco = json.load(f)

coco_cats = {c["id"]: c["name"] for c in coco["categories"]}
print(f"✅ Loaded {len(coco_cats)} classes from COCO")

# Load ODVG
odvg_cats = set()
with jsonlines.open(odvg_path) as reader:
    for ex in reader:
        if "detection" in ex:
            dets = ex["detection"]
            if isinstance(dets, dict) and "instances" in dets:
                for inst in dets["instances"]:
                    if "category" in inst:
                        odvg_cats.add(inst["category"])
            elif isinstance(dets, list):
                for d in dets:
                    if isinstance(d, dict) and "category" in d:
                        odvg_cats.add(d["category"])
                    elif isinstance(d, str):
                        odvg_cats.add(d)
print(f"✅ Found {len(odvg_cats)} unique categories in ODVG")

# Cross-check
print("\n--- Cross-check IDs ---")

# 1. Cek ID di label_map
expected_ids = set(range(len(id2cat)))
if set(id2cat.keys()) != expected_ids:
    print(f"❌ Label map IDs salah. Harus {expected_ids}, tapi ada {set(id2cat.keys())}")
else:
    print("✅ Label map IDs urut mulai dari 0")

# 2. Cek konsistensi nama label vs COCO
missing_in_coco = set(id2cat.values()) - set(coco_cats.values())
extra_in_coco = set(coco_cats.values()) - set(id2cat.values())

if missing_in_coco:
    print(f"⚠️ Ada kelas di label_map yang hilang di COCO: {missing_in_coco}")
if extra_in_coco:
    print(f"⚠️ Ada kelas di COCO yang ga ada di label_map: {extra_in_coco}")
if not missing_in_coco and not extra_in_coco:
    print("✅ Nama kelas COCO = label_map")

# 3. Cek apakah index di COCO sesuai mapping
mismatch_ids = []
for cid, cname in coco_cats.items():
    if cname in cat2id and cat2id[cname] != cid:
        mismatch_ids.append((cname, cid, cat2id[cname]))

if mismatch_ids:
    print("❌ ID mismatch:")
    for cname, coco_id, label_id in mismatch_ids:
        print(f"   - {cname}: COCO id={coco_id}, label_map id={label_id}")
else:
    print("✅ COCO IDs cocok dengan label_map")

# 4. Cek ODVG consistency
missing_in_odvg = set(id2cat.values()) - odvg_cats
extra_in_odvg = odvg_cats - set(id2cat.values())

if missing_in_odvg:
    print(f"⚠️ Ada kelas di label_map yang ga muncul di ODVG: {missing_in_odvg}")
if extra_in_odvg:
    print(f"⚠️ Ada kelas di ODVG yang ga ada di label_map: {extra_in_odvg}")
if not missing_in_odvg and not extra_in_odvg:
    print("✅ ODVG classes cocok dengan label_map")

print("\nCheck selesai 🚀")
