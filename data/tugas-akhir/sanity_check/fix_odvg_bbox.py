import jsonlines

input_file = "annotations/train_odvg_clean.jsonl"
output_file = "annotations/train_odvg_fixed.jsonl"

fixed_count = 0

with jsonlines.open(input_file, "r") as reader, jsonlines.open(output_file, "w") as writer:
    for obj in reader:
        img_w, img_h = obj.get("width"), obj.get("height")
        annotations = obj.get("annotations", [])

        for ann in annotations:
            if "bbox" in ann:
                x, y, w, h = ann["bbox"]

                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)

                new_bbox = [round(float(x), 2), round(float(y), 2),
                            round(float(w), 2), round(float(h), 2)]

                if new_bbox != ann["bbox"]:
                    fixed_count += 1
                    ann["bbox"] = new_bbox

        writer.write(obj)

print(f"🔧 Total bbox diperbaiki di ODVG: {fixed_count}")
print(f"✅ File sudah disimpan ke: {output_file}")
