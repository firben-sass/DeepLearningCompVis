import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

RESIZE_W, RESIZE_H = 600, 600
IOU_THRESH = 0.5
MAX_PROPOSALS = 1000

model_path = "/Users/dani/Desktop/gitt/DeepLearningCompVis/projects/boundingBox/Task_4.1/model.yml"

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter + 1e-6

    return inter / union

def parse_gt_boxes_resized(xml_path, resize_w, resize_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w_orig = float(root.find("size/width").text)
    h_orig = float(root.find("size/height").text)

    scale_x = resize_w / w_orig
    scale_y = resize_h / h_orig

    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        xmin = int(float(b.find("xmin").text) * scale_x)
        ymin = int(float(b.find("ymin").text) * scale_y)
        xmax = int(float(b.find("xmax").text) * scale_x)
        ymax = int(float(b.find("ymax").text) * scale_y)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes



def get_proposals_for_image(image_path, xml_path=None, iou_thresh=0.5, top_k=None):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")


    img_resized = cv2.resize(img, (RESIZE_W, RESIZE_H))

    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(MAX_PROPOSALS)

    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    edges = edge_detector.detectEdges(rgb)
    orientation = edge_detector.computeOrientation(edges)
    edges = edge_detector.edgesNms(edges, orientation)
    boxes_xywh, scores = edge_boxes.getBoundingBoxes(edges, orientation)
    boxes_xywh = np.array(boxes_xywh)

    if top_k is not None:
        boxes_xywh = boxes_xywh[:top_k]

    boxes_xyxy = np.array([xywh_to_xyxy(b) for b in boxes_xywh])

    #ground truth boxes
    tree = ET.parse(xml_path)
    root = tree.getroot()
    w_orig = float(root.find("size/width").text)
    h_orig = float(root.find("size/height").text)
    gt_boxes = parse_gt_boxes_resized(xml_path, RESIZE_W, RESIZE_H)

    #Iou calculation and filtering
    iou_scores = []
    filtered_boxes = []

    for p in boxes_xyxy:
        max_iou = max(iou(p, gt) for gt in gt_boxes)
        iou_scores.append(max_iou)
        if max_iou >= iou_thresh:
            filtered_boxes.append(p)

    return filtered_boxes, boxes_xyxy, iou_scores

image_name = "potholes12.png"

image_path = os.path.join("potholes/images", image_name)
xml_path = os.path.join("potholes/annotations", image_name.replace(".png", ".xml"))

filtered, all_props, scores = get_proposals_for_image(
    image_path,
    xml_path=xml_path,
    iou_thresh=0.5,
    top_k=1000
)

print("Total proposals:", len(all_props))
print("Passing IoU threshold:", len(filtered))
