import os

import matplotlib.pyplot as plt
import numpy as np
from boundingBox.lib.tools.iou import IOU


def _ensure_ndarray(values):
    """Convert list-like data to a float ndarray while preserving length checks."""
    return np.asarray(values, dtype=float) if len(values) else np.zeros((0,))


def _match_detections(boxes, scores, true_boxes, IOU_threshold):
    """Greedily assign detections to ground-truth boxes in score order."""
    boxes = _ensure_ndarray(boxes)
    scores = _ensure_ndarray(scores)
    true_boxes = _ensure_ndarray(true_boxes)

    if boxes.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    order = np.argsort(scores)[::-1]
    matched_true = np.zeros(len(true_boxes), dtype=bool)
    true_positive = np.zeros(len(order), dtype=float)
    false_positive = np.zeros(len(order), dtype=float)

    for rank, det_idx in enumerate(order):
        if len(true_boxes) == 0:
            false_positive[rank] = 1.0
            continue

        detection_box = boxes[det_idx]
        ious = np.array([IOU(detection_box, gt_box) for gt_box in true_boxes])
        best_gt = np.argmax(ious)
        best_iou = ious[best_gt]

        if best_iou >= IOU_threshold and not matched_true[best_gt]:
            true_positive[rank] = 1.0
            matched_true[best_gt] = True
        else:
            false_positive[rank] = 1.0

    return true_positive, false_positive

def average_precision(boxes, scores, true_boxes, IOU_threshold):
    boxes = _ensure_ndarray(boxes)
    scores = _ensure_ndarray(scores)
    true_boxes = _ensure_ndarray(true_boxes)
    num_true = len(true_boxes)

    if num_true == 0:
        return 0.0, np.zeros((1, 2))

    tp, fp = _match_detections(boxes, scores, true_boxes, IOU_threshold)
    if tp.size == 0:
        recall = np.array([0.0])
        precision = np.array([0.0])
    else:
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / num_true
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = np.trapz(precision, recall)
    rp = np.column_stack((recall, precision))

    return ap, rp

def precision_recall_curve(boxes, scores, true_boxes, IOU_threshold, save_path):
    """Generate and save a precision-recall curve for the given detections."""
    average_precision_value, recall_precision = average_precision(
        boxes,
        scores,
        true_boxes,
        IOU_threshold,
    )

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

