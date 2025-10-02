import json
import jsonlines
import os

# Paths
coco_src = "../valid/_annotations.coco_fixed.json"
coco_out = "../valid/_annotations.coco_fixed.json"
odvg_src = "../annotations/train_odvg_fixed.jsonl"
odvg_out = "../annotations/train_odvg_fixed.jsonl"

def fix_coco(src, dst):
    with open(src, "r") as f:
        data = json.load(f)

    fixed_ann = []
    dropped = 0

    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        img_id = ann["image_id"]

        # Normalisasi: pastikan w,h > 0
        if w <= 0 or h <= 0:
            dropped += 1
            continue

        # Clamp ke boundary gambar (kalau ada info ukuran)
        # Default pakai None, dicek dulu
        img_info = next((img for img in data["images"] if img["id"] == img_id), None)
        if img_info:
            W, H = img_info["width"], img_info["height"]
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = min(w, W - x)
            h = min(h, H - y)

        ann["bbox"] = [x, y, w, h]
        fixed_ann.append(ann)

    data["annotations"] = fixed_ann

    with open(dst, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ COCO fixed: {len(fixed_ann)} annos (dropped {dropped})")
    return dst


def fix_odvg(src, dst):
    fixed_items = []
    dropped = 0

    with jsonlines.open(src) as reader:
        for ex in reader:
            if "detection" in ex:
                dets = ex["detection"]

                # ODVG bisa 3 case
                if isinstance(dets, dict) and "instances" in dets:
                    new_instances = []
                    for d in dets["instances"]:
                        if "bbox" in d:
                            x1, y1, x2, y2 = d["bbox"]

                            # Swap kalau ketuker
                            if x2 < x1:
                                x1, x2 = x2, x1
                            if y2 < y1:
                                y1, y2 = y2, y1

                            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                                dropped += 1
                                continue

                            d["bbox"] = [x1, y1, x2, y2]
                            new_instances.append(d)

                    if new_instances:
                        ex["detection"]["instances"] = new_instances
                        fixed_items.append(ex)

                elif isinstance(dets, list) and all(isinstance(d, dict) for d in dets):
                    new_dets = []
                    for d in dets:
                        if "bbox" in d:
                            x1, y1, x2, y2 = d["bbox"]
                            if x2 < x1:
                                x1, x2 = x2, x1
                            if y2 < y1:
                                y1, y2 = y2, y1
                            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                                dropped += 1
                                continue
                            d["bbox"] = [x1, y1, x2, y2]
                            new_dets.append(d)
                    if new_dets:
                        ex["detection"] = new_dets
                        fixed_items.append(ex)

                else:
                    fixed_items.append(ex)

    with jsonlines.open(dst, "w") as writer:
        writer.write_all(fixed_items)

    print(f"✅ ODVG fixed: {len(fixed_items)} entries (dropped {dropped})")
    return dst


if __name__ == "__main__":
    print("=== Fixing BBoxes COCO & ODVG ===")
    fix_coco(coco_src, coco_out)
    fix_odvg(odvg_src, odvg_out)
    print("Selesai 🚀")
