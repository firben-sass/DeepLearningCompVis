import os

import matplotlib.pyplot as plt
import numpy as np
from boundingBox.lib.tools.iou import IOU


def match_detections(boxes, scores, gt, IOU_threshold):
    confidence_order = np.argsort(scores)[::-1]
    boxes = boxes[confidence_order]
    scores = scores[confidence_order]
    matched_gt = np.full(gt.shape[0], False)
    tp_fp = []

    for i, box in enumerate(boxes):
        ious = []
        for j, gt_box in enumerate(gt):
            iou = IOU(box, gt_box)
            ious.append(iou)
        ious = np.array(ious)
        ious[matched_gt] = 0
        max_match = np.argmax(ious)
        if ious[max_match] > IOU_threshold:
            matched_gt[max_match] = True
            tp_fp.append([scores[i], 1])
        else:
            tp_fp.append([scores[i], 0])

    return np.asarray(tp_fp)


def match_detections_across_dataset(boxes, scores, gt, IOU_threshold):
    tp_fp_blocks = []
    for i, boxes_ in enumerate(boxes):
        tp_fp_blocks.append(match_detections(boxes_, scores[i], gt[i], IOU_threshold))
    tp_fp = np.vstack(tp_fp_blocks)
    tp_fp = tp_fp[tp_fp[:, 0].argsort()]
    return tp_fp


def average_precision(boxes, scores, gt, IOU_threshold):
    tp_fp = match_detections_across_dataset(boxes, scores, gt, IOU_threshold)
    tp_count = 0.0
    fp_count = 0.0
    precisions = []
    recalls = [0]
    total_positives = np.sum([len(gt_) for gt_ in gt])
    for k in range(len(tp_fp)):
        if tp_fp[k,1]:
            tp_count += 1.0
        else:
            fp_count += 1.0
        precisions.append(tp_count / (k+1))
        recalls.append(tp_count / total_positives)
    
    ap = np.sum([(recalls[i+1] - recalls[i]) * precisions[i] for i in range(len(precisions))])
    return ap, recalls, precisions


def precision_recall_curve(average_precision_value, recalls, precisions, save_path):
    """Generate and save a precision-recall curve for the given detections."""

    recall_precision = np.array(list(zip(recalls[1:], precisions)))

    if recall_precision.size == 0:
        print("Warning: recall/precision array is empty; cannot plot curve.")
        raise ValueError("Precision/recall array is empty; cannot plot curve.")

    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        recall_precision[:, 0],
        recall_precision[:, 1],
        marker="o",
        label=f"AP = {average_precision_value:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)