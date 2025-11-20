import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET


images_folder    = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/images"
xml_folder       = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/annotations"
proposals_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/proposals"
model_path       = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/model.yml"

os.makedirs(proposals_folder, exist_ok=True)


RESIZE_W, RESIZE_H = 600, 600
IOU_THRESH = 0.5
RUN_PROPOSALS = False

#

def xywh_to_xyxy(box):
    """
    Convert [x, y, w, h] (top-left + width/height) to
    [xmin, ymin, xmax, ymax] (two corners).
    """
    x, y, w, h = box
    return [x, y, x + w, y + h]

def iou(box1, box2):
    """
    Compute IoU between two boxes in [xmin, ymin, xmax, ymax] format.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    # Areas
    area1 = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    area2 = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)

    union = area1 + area2 - inter_area + 1e-6  
    return inter_area / union

def parse_gt_boxes_resized(xml_path, w_orig, h_orig, resize_w, resize_h):
    """
    Parse ground-truth boxes from a VOC-style XML file and
    rescale them from (w_orig, h_orig) to (resize_w, resize_h).
    Returns list of [xmin, ymin, xmax, ymax] in resized coordinates.
    """
    scale_x = resize_w / float(w_orig)
    scale_y = resize_h / float(h_orig)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))

        # Scale to resized coordinates
        xmin_r = int(round(xmin * scale_x))
        xmax_r = int(round(xmax * scale_x))
        ymin_r = int(round(ymin * scale_y))
        ymax_r = int(round(ymax * scale_y))

        gt_boxes.append([xmin_r, ymin_r, xmax_r, ymax_r])

    return gt_boxes

#EdgeBoxes

image_files = [f for f in os.listdir(images_folder)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# if RUN_PROPOSALS:
#     print("Running EdgeBoxes proposal generation...")
#     # Initialize EdgeBoxes
#     edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)
#     edge_boxes = cv2.ximgproc.createEdgeBoxes()
#     edge_boxes.setMaxBoxes(2000)   # adjust if you want

#     for img_name in image_files:
#         img_path = os.path.join(images_folder, img_name)
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Could not read {img_path}, skipping.")
#             continue

#         # Resize to fixed size
#         img_resized = cv2.resize(img, (RESIZE_W, RESIZE_H))

#         # Compute edges + orientation
#         rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
#         edges = edge_detector.detectEdges(rgb)
#         orientation = edge_detector.computeOrientation(edges)
#         edges = edge_detector.edgesNms(edges, orientation)

#         # EdgeBoxes proposals
#         boxes, scores = edge_boxes.getBoundingBoxes(edges, orientation)
#         # boxes is list of [x, y, w, h]

#         # Save bounding boxes as .npy
#         np.save(os.path.join(proposals_folder, img_name + ".npy"), np.array(boxes))

#         # Optional: visualize top 50 proposals
#         vis = img_resized.copy()
#         for (x, y, w, h) in boxes[:50]:
#             cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

#         cv2.imwrite(os.path.join(proposals_folder, img_name), vis)

#     print("EdgeBoxes proposals saved to:", proposals_folder)

#Evaliuation using IoU > K

# print("\nEvaluating proposals against ground truth (IoU > {:.2f})...".format(IOU_THRESH))

# total_gt = 0
# total_hits = 0

# for img_name in image_files:
#     img_path = os.path.join(images_folder, img_name)
#     xml_path = os.path.join(xml_folder, os.path.splitext(img_name)[0] + ".xml")
#     prop_path = os.path.join(proposals_folder, img_name + ".npy")

#     if not os.path.exists(xml_path):
#         print(f"[WARN] No XML for {img_name}, skipping.")
#         continue
#     if not os.path.exists(prop_path):
#         print(f"[WARN] No proposals for {img_name}, skipping.")
#         continue

#     # Load original image to get original size
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[WARN] Could not read {img_path}, skipping.")
#         continue
#     h_orig, w_orig = img.shape[:2]

#     # Parse and resize GT boxes
#     gt_boxes_resized = parse_gt_boxes_resized(xml_path, w_orig, h_orig, RESIZE_W, RESIZE_H)
#     if len(gt_boxes_resized) == 0:
#         print(f"[INFO] No GT boxes for {img_name}, skipping.")
#         continue

#     # Load proposals and convert to [xmin, ymin, xmax, ymax]
#     boxes = np.load(prop_path)  # shape [N, 4], [x, y, w, h]
#     proposals_xyxy = [xywh_to_xyxy(b) for b in boxes]

#     hits_img = 0
#     for gt in gt_boxes_resized:
#         if len(proposals_xyxy) == 0:
#             best_iou = 0.0
#         else:
#             best_iou = max(iou(gt, p) for p in proposals_xyxy)

#         if best_iou >= IOU_THRESH:
#             hits_img += 1

#     total_gt += len(gt_boxes_resized)
#     total_hits += hits_img

#     recall_img = hits_img / float(len(gt_boxes_resized))
#     print(f"{img_name}: GT={len(gt_boxes_resized)}, hits={hits_img}, recall={recall_img:.3f}")

# overall_recall = total_hits / float(total_gt) if total_gt > 0 else 0.0
# print("\n==============================")
# print("Overall recall (IoU > {:.2f}): {:.3f}".format(IOU_THRESH, overall_recall))
# print("Total GT boxes:", total_gt)
# print("Total hits:    ", total_hits)
# print("==============================")


##
# Recall calulations
##
Ks = [10, 20, 50, 100, 200, 500, 1000, 2000]  # adjust to your max boxes

print("\nEvaluating recall for different numbers of proposals (IoU > {:.2f})".format(IOU_THRESH))

for K in Ks:
    total_gt = 0
    total_hits = 0

    for img_name in image_files:
        img_path = os.path.join(images_folder, img_name)
        xml_path = os.path.join(xml_folder, os.path.splitext(img_name)[0] + ".xml")
        prop_path = os.path.join(proposals_folder, img_name + ".npy")

        if not os.path.exists(xml_path) or not os.path.exists(prop_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]

        gt_boxes_resized = parse_gt_boxes_resized(xml_path, w_orig, h_orig, RESIZE_W, RESIZE_H)
        if len(gt_boxes_resized) == 0:
            continue

        # Load proposals, keep only top K
        boxes = np.load(prop_path)  # [N,4] in [x,y,w,h]
        boxes = boxes[:K]           # TOP K proposals
        proposals_xyxy = [xywh_to_xyxy(b) for b in boxes]

        hits_img = 0
        for gt in gt_boxes_resized:
            if len(proposals_xyxy) == 0:
                best_iou = 0.0
            else:
                best_iou = max(iou(gt, p) for p in proposals_xyxy)

            if best_iou >= IOU_THRESH:
                hits_img += 1

        total_gt += len(gt_boxes_resized)
        total_hits += hits_img

    overall_recall = total_hits / float(total_gt) if total_gt > 0 else 0.0
    print(f"K = {K:4d}  ->  recall = {overall_recall:.3f}")
