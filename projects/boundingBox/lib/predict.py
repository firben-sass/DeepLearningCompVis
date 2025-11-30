import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torchvision.transforms import functional as F

from .model.mobilnet import load_proposal_classifier
from .tools.proposals import xywh_to_xyxy, parse_gt_boxes_resized
from .tools.images import load_and_resize_image, load_images, extract_image_patches
from .tools.iou import IOU
from .tools.visualisation import visualize_bboxes_with_patches
from .tools.load_bounding_boxes import load_bounding_boxes
from .nms import nms
from .average_precision import average_precision, precision_recall_curve


def get_probs(patches, model, resize_img, batch_size=64):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    probabilities = []
    batch_tensors = []

    # Process patches in manageable batches to limit GPU memory use
    for patch in patches:
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(patch_rgb)
        tensor = F.resize(tensor, resize_img)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        batch_tensors.append(tensor)

        if len(batch_tensors) == batch_size:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                logits = model(batch)
            logits = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(logits.cpu().tolist())
            batch_tensors.clear()

    if batch_tensors:
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
        logits = torch.softmax(logits, dim=1)[:, 1]
        probabilities.extend(logits.cpu().tolist())

    if was_training:
        model.train()

    return probabilities


def get_proposals_for_image(image, top_k=None, max_proposals=1000):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "model.yml")

    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(MODEL_PATH)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(max_proposals)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    edges = edge_detector.detectEdges(rgb)
    orientation = edge_detector.computeOrientation(edges)
    edges = edge_detector.edgesNms(edges, orientation)
    boxes_xywh, scores = edge_boxes.getBoundingBoxes(edges, orientation)
    scores = scores.flatten()
    boxes_xywh = boxes_xywh[np.argsort(-np.array(scores))]

    if top_k is not None:
        boxes_xywh = boxes_xywh[:top_k]

    boxes_xyxy = np.array([xywh_to_xyxy(boxes_xywh[i,:]) for i in range(len(boxes_xywh))])

    return boxes_xyxy


def get_image_and_xml_paths(split_info_path):
    data = np.load(split_info_path, allow_pickle=True)

    # print(len(data.item(0)["train_images"]))
    # print(len(data.item(0)["val_images"]))
    image_paths = ["boundingBox/data/images/" + str(elem) for elem in data.item(0)["test_images"]]
    xml_paths = [path.replace("images", "annotations").replace(".png", ".xml") for path in image_paths]
    return image_paths, xml_paths


def main():
    iou_thresh = 0.4
    confidence_thresh = 0.7
    top_k = 300
    resize_img = (600, 600)
    resize_patches = (224, 224)
    max_proposals = 1000

    split_info_path = '/work3/s204164/DeepLearningCompVis/projects/boundingBox/Task_4.1/coord_proposals/dataset_crops_split/split_info.npy'
    image_paths, xml_paths = get_image_and_xml_paths(split_info_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_proposal_classifier(
        weights_path="/work3/s204164/DeepLearningCompVis/projects/boundingBox/models/proposal_classifier_10.pth",
        num_classes=2,
        device=device,
        pretrained=False,
        strict=True
    )

    images = load_images(image_paths, resize_img)

    all_filtered_boxes = []
    all_filtered_probs = []
    all_true_boxes = []

    for index, image in enumerate(images):
        boxes_xyxy = get_proposals_for_image(image, top_k=top_k, max_proposals=max_proposals)
        # print("boxes_xyxy len:", len(boxes_xyxy))
        # print("First coords:", boxes_xyxy[0])
        patches = extract_image_patches(image, boxes_xyxy)
        probs = get_probs(patches, model, resize_patches, batch_size=64)
        probs = np.array(probs)
        # print("Positive probs: ", probs[probs > 0.5])

        # visualize_bboxes_with_patches(
        #     image=images[0],
        #     boxes=boxes_xyxy,
        #     probabilities=probs,
        #     output_path="/work3/s204164/DeepLearningCompVis/projects/boundingBox/outputs/annotated_image.png",
        #     color=(0, 255, 0),
        #     thickness=2,
        #     show_indices=True,
        #     figsize=(10, 10),
        # )

        true_boxes = load_bounding_boxes(xml_paths[index], resize_img)
        all_true_boxes.extend(true_boxes)

        filtered_boxes, filtered_probs = nms(
            boxes_xyxy,
            probs,
            confidence_threshold=confidence_thresh,
            IOU_threshold=iou_thresh,
        )
        if len(filtered_boxes) == 0:
            continue

        all_filtered_boxes.append(filtered_boxes)
        all_filtered_probs.append(filtered_probs)
        # visualize_bboxes_with_patches(
        #     image=images[0],
        #     boxes=filtered_boxes,
        #     probabilities=filtered_probs,
        #     output_path="/work3/s204164/DeepLearningCompVis/projects/boundingBox/outputs/annotated_image.png",
        #     gt_xml_path=xml_paths[index],
        #     color=(0, 255, 0),
        #     thickness=2,
        #     show_indices=True,
        #     figsize=(10, 10),
        # )

    aggregated_boxes = np.vstack(all_filtered_boxes) if all_filtered_boxes else np.zeros((0, 4))
    aggregated_probs = (
        np.concatenate(all_filtered_probs)
        if all_filtered_probs
        else np.zeros(0, dtype=float)
    )
    aggregated_true = (
        np.asarray(all_true_boxes, dtype=float)
        if all_true_boxes
        else np.zeros((0, 4))
    )

    average_precision_value, recall_precision = average_precision(
        aggregated_boxes,
        aggregated_probs,
        aggregated_true,
        iou_thresh,
    )
    print("Average Precision:", average_precision_value)
    precision_recall_curve(
        aggregated_boxes,
        aggregated_probs,
        aggregated_true,
        iou_thresh,
        save_path="/work3/s204164/DeepLearningCompVis/projects/boundingBox/outputs/precision_recall_curve.png",
    )

if __name__ == "__main__":
    main()