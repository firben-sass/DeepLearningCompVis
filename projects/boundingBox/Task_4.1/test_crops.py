import cv2
import os
import numpy as np
import csv


images_folder    = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/images"
proposals_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/proposals"

RESIZE_W, RESIZE_H = 600, 600

TEST_IMAGE = "potholes0.png"
NUM_CROPS  = 10

csv_path = os.path.join(proposals_folder, "train_samples.csv")

output_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/debug_crops"
os.makedirs(output_folder, exist_ok=True)


labels_dict = {}

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["image"] == TEST_IMAGE:
            x = int(float(row["x"]))
            y = int(float(row["y"]))
            w = int(float(row["w"]))
            h = int(float(row["h"]))
            label = int(row["label"])
            labels_dict[(x, y, w, h)] = label


img_path  = os.path.join(images_folder, TEST_IMAGE)
prop_path = os.path.join(proposals_folder, TEST_IMAGE + ".npy")

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")
if not os.path.exists(prop_path):
    raise FileNotFoundError(f"Proposals not found: {prop_path}")

img = cv2.imread(img_path)
if img is None:
    raise RuntimeError(f"Could not read image: {img_path}")


img_resized = cv2.resize(img, (RESIZE_W, RESIZE_H))

boxes = np.load(prop_path)  # [N, 4] in [x, y, w, h]

print(f"Total proposals for {TEST_IMAGE}: {len(boxes)}")

for i, (x, y, w, h) in enumerate(boxes[:NUM_CROPS]):
    x, y, w, h = int(x), int(y), int(w), int(h)

    # get label or mark as unknown
    label = labels_dict.get((x, y, w, h), -1)  # -1 = no label found (likely ambiguous IoU)

    x2 = min(x + w, RESIZE_W)
    y2 = min(y + h, RESIZE_H)
    x  = max(x, 0)
    y  = max(y, 0)
    crop = img_resized[y:y2, x:x2]

    out_name = f"{os.path.splitext(TEST_IMAGE)[0]}_crop_{i}_label_{label}.png"
    out_path = os.path.join(output_folder, out_name)
    cv2.imwrite(out_path, crop)

    print(f"Crop {i}: location=({x},{y},{w},{h}), label={label} -> saved {out_path}")

print("\nDone. Open the debug crops in:", output_folder)
