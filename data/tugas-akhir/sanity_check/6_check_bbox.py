import json
import jsonlines
import os

# Paths
coco_path = "../valid/_annotations.coco_fixed.json"
odvg_path = "../annotations/train_odvg_fixed.jsonl"

def check_coco(path):
    print(f"\n=== Checking COCO: {os.path.basename(path)} ===")
    with open(path, "r") as f:
        data = json.load(f)

    n_cat = len(data.get("categories", []))
    n_ann = len(data.get("annotations", []))
    n_img = len(data.get("images", []))
    print(f"Jumlah kategori: {n_cat}")
    print(f"Jumlah anotasi : {n_ann}")
    print(f"Jumlah gambar  : {n_img}")

    bad = []
    for ann in data["annotations"]:
        bbox = ann["bbox"]  # [x, y, w, h]
        if bbox[2] <= 0 or bbox[3] <= 0:
            bad.append(("zero", bbox, ann["image_id"]))

    if bad:
        print(f"⚠️ Ada {len(bad)} bbox dengan width/height nol")
    else:
        print("✅ Tidak ada bbox berukuran nol.")

def check_odvg(path):
    print(f"\n=== Checking ODVG: {os.path.basename(path)} ===")
    with jsonlines.open(path) as reader:
        items = list(reader)

    n_items = len(items)
    print(f"Jumlah data: {n_items}")

    bad = []
    for i, ex in enumerate(items):
        if "detection" in ex and "instances" in ex["detection"]:
            for det in ex["detection"]["instances"]:
                bbox = det.get("bbox", None)
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    if x2 <= x1 or y2 <= y1:
                        bad.append((i, bbox, ex.get("filename", "?")))

    if bad:
        print(f"⚠️ Ada {len(bad)} bbox tidak valid (x2<=x1 atau y2<=y1)")
        for b in bad[:10]:
            print("   ", b)
    else:
        print("✅ Semua bbox valid di ODVG")

if __name__ == "__main__":
    check_coco(coco_path)
    check_odvg(odvg_path)
    print("\nCek selesai 🚀")
