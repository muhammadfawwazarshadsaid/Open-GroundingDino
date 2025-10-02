import json
import jsonlines
import os

odvg_path = "../annotations/train_odvg.jsonl"
coco_path = "../valid/_annotations.coco.json"
labelmap_path = "../config/label_map.json"

def check_labelmap(labelmap_path):
    with open(labelmap_path, "r") as f:
        labelmap = json.load(f)
    print(f"Jumlah kelas di label_map.json: {len(labelmap)}")
    print("Contoh mapping:", list(labelmap.items())[:5])

    if "0" not in labelmap:
        print("❌ ERROR: label_map tidak mulai dari 0")
    else:
        print("✅ label_map mulai dari 0")
    return set(labelmap.values())

def check_odvg(odvg_path, classes):
    with jsonlines.open(odvg_path) as reader:
        for i, obj in enumerate(reader):
            if "filename" not in obj:
                print(f"❌ Baris {i}: missing filename")
            if "detection" not in obj:
                print(f"❌ Baris {i}: missing detection")
            else:
                for inst in obj["detection"]["instances"]:
                    if "bbox" not in inst or len(inst["bbox"]) != 4:
                        print(f"❌ Baris {i}: bbox invalid {inst}")
                    if "category" not in inst:
                        print(f"❌ Baris {i}: missing category")
                    elif inst["category"] not in classes:
                        print(f"⚠️ Baris {i}: category '{inst['category']}' tidak ada di label_map.json")
            if i > 50: 
                break
    print("✅ Format ODVG cek sebagian aman.")

def check_coco(coco_path, classes):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    annos = coco.get("annotations", [])
    cats = {c["name"] for c in coco.get("categories", [])}

    print(f"Jumlah kategori di COCO: {len(cats)}")
    print(f"Jumlah anotasi di COCO: {len(annos)}")

    for i, a in enumerate(annos[:50]):
        if "bbox" not in a or len(a["bbox"]) != 4:
            print(f"❌ Anotasi {i}: bbox invalid {a}")
    print("✅ Format COCO cek sebagian aman.")

if __name__ == "__main__":
    classes = check_labelmap(labelmap_path)
    check_odvg(odvg_path, classes)
    check_coco(coco_path, classes)
