import json
import jsonlines

def check_coco(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"=== Checking COCO: {file_path} ===")
    w_range_error = 0
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        img = next(img for img in data["images"] if img["id"] == ann["image_id"])
        img_w, img_h = img["width"], img["height"]

        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            w_range_error += 1

    if w_range_error == 0:
        print("✅ Semua bbox valid di COCO")
    else:
        print(f"⚠️ Masih ada {w_range_error} bbox keluar batas di COCO")


def check_odvg(file_path):
    with jsonlines.open(file_path, "r") as reader:
        print(f"=== Checking ODVG: {file_path} ===")
        w_range_error = 0
        for obj in reader:
            img_w, img_h = obj.get("width"), obj.get("height")
            for ann in obj.get("annotations", []):
                if "bbox" in ann:
                    x, y, w, h = ann["bbox"]
                    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                        w_range_error += 1

        if w_range_error == 0:
            print("✅ Semua bbox valid di ODVG")
        else:
            print(f"⚠️ Masih ada {w_range_error} bbox keluar batas di ODVG")


check_coco("valid/_annotations.coco_fixed.json")
check_odvg("annotations/train_odvg_fixed.jsonl")
