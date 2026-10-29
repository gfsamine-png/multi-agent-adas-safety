import json
from collections import defaultdict

boxes = defaultdict(list)
n = 0
with open("outputs/observations.jsonl") as f:
    for line in f:
        obs = json.loads(line)
        n += 1
        for d in obs["objects"]:
            if d["class_name"] == "person":
                boxes[d.get("track_id")].append(d["bbox"])

print("frames:", n)
for tid, bs in sorted(boxes.items(), key=lambda kv: -len(kv[1])):
    k = len(bs)
    x1 = sum(b[0] for b in bs) / k
    y1 = sum(b[1] for b in bs) / k
    x2 = sum(b[2] for b in bs) / k
    y2 = sum(b[3] for b in bs) / k
    print(f"id {tid}: {k} frames, avg box x {x1:.0f}-{x2:.0f}, y {y1:.0f}-{y2:.0f}")