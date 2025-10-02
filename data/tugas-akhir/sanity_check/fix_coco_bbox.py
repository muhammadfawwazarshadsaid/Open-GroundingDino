import json

val_coco = "valid/_annotations.coco_clean.json"
output_file = "valid/_annotations.coco_fixed.json"

with open(val_coco, "r") as f:
    data = json.load(f)

image_dict = {img["id"]: img for img in data["images"]}
fixed_count = 0

for ann in data["annotations"]:
    bbox = ann["bbox"] 
    img = image_dict[ann["image_id"]]
    img_w, img_h = img["width"], img["height"]

    x, y, w, h = bbox

    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    new_bbox = [round(float(x), 2), round(float(y), 2),
                round(float(w), 2), round(float(h), 2)]

    if new_bbox != bbox:
        fixed_count += 1
        ann["bbox"] = new_bbox

print(f"🔧 Total bbox diperbaiki: {fixed_count}")

with open(output_file, "w") as f:
    json.dump(data, f)

print(f"✅ File sudah disimpan ke: {output_file}")
