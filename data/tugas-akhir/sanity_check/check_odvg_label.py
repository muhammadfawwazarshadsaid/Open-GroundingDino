import json
import jsonlines

odvg_path = "../annotations/train_odvg_final.jsonl"
labelmap_path = "../config/label_map_final.json"

# load labelmap
with open(labelmap_path) as f:
    labelmap = json.load(f)
valid_ids = set(map(int, labelmap.keys()))

bad = []
with jsonlines.open(odvg_path) as reader:
    for i, ex in enumerate(reader):
        if "detection" in ex:
            dets = ex["detection"]
            if isinstance(dets, dict) and "instances" in dets:
                for det in dets["instances"]:
                    if "label" in det and det["label"] not in valid_ids:
                        bad.append((i, det))
            elif isinstance(dets, list):
                for det in dets:
                    if "label" in det and det["label"] not in valid_ids:
                        bad.append((i, det))

print(f"🚨 Found {len(bad)} invalid labels")
for i, det in bad[:10]:  # print 10 sample pertama
    print(f"Line {i} → {det}")
