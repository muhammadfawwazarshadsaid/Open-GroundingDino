import json
import jsonlines
import os

odvg_path = "../annotations/train_odvg.jsonl"
coco_path = "../valid/_annotations.coco.json"
labelmap_path = "../config/label_map.json"

def fix_labelmap(path):
    with open(path, "r") as f:
        data = json.load(f)

    # key diurutkan
    items = list(data.items())

    # kalau index ga mulai dari 0 → reindex ulang
    keys = [int(k) for k, _ in items]
    if min(keys) != 0:
        print("❌ Label map tidak mulai dari 0 → auto-fix ...")
        fixed = {str(i): v for i, (_, v) in enumerate(items)}
        with open(path.replace(".json", "_fixed.json"), "w") as f:
            json.dump(fixed, f, indent=2)
        print(f"✅ label_map fixed → {os.path.basename(path.replace('.json', '_fixed.json'))}")
        return path.replace(".json", "_fixed.json")
    else:
        print("✅ Label map sudah valid (mulai dari 0)")
        return path

# Sanity check ODVG format
def check_odvg(path):
    with jsonlines.open(path) as reader:
        for i, ex in enumerate(reader):
            if "detection" not in ex:
                print(f"⚠️ Line {i}: ga ada field 'detection'")
            if "filename" not in ex:
                print(f"⚠️ Line {i}: ga ada field 'filename'")
            if i < 3:
                print(f"🔎 Sample {i+1}: keys = {list(ex.keys())}")
    print("✅ ODVG basic format checked.")

# Sanity check COCO format
def check_coco(path):
    with open(path, "r") as f:
        data = json.load(f)

    if "categories" not in data or "annotations" not in data:
        print("❌ COCO format error: missing keys")
        return

    cat_ids = [c["id"] for c in data["categories"]]
    if min(cat_ids) != 0:
        print("⚠️ Category ID di COCO tidak mulai dari 0 (tidak fatal, tapi cek konsistensi).")

    print(f"✅ COCO format oke. Categories = {len(data['categories'])}, Annotations = {len(data['annotations'])}")

print("\n=== Fixing Dataformat ===")
fixed_labelmap = fix_labelmap(labelmap_path)
check_odvg(odvg_path)
check_coco(coco_path)

print("\nSelesai ✅\nGunakan:", fixed_labelmap, "untuk update config kalau file baru dibuat.")
