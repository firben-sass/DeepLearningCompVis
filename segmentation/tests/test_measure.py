import pytest
import torch

from segmentation import measure


def test_flatten_masks_shape_mismatch():
    pred = torch.zeros((1, 1, 2, 2))
    target = torch.zeros((1, 1, 2, 3))
    with pytest.raises(ValueError, match="pred and target must have the same shape"):
        measure._flatten_masks(pred, target)


def test_flatten_masks_invalid_dims():
    pred = torch.zeros((1, 1, 2))
    target = torch.zeros((1, 1, 2))
    with pytest.raises(ValueError, match=r"shape \(B, C, H, W\)"):
        measure._flatten_masks(pred, target)


def test_metrics_perfect_overlap():
    pred = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.float32)
    target = pred.clone()

    assert measure.dice_overlap(pred, target) == pytest.approx(1.0)
    assert measure.intersection_over_union(pred, target) == pytest.approx(1.0)
    assert measure.accuracy(pred, target) == pytest.approx(1.0)
    assert measure.sensitivity(pred, target) == pytest.approx(1.0)
    assert measure.specificity(pred, target) == pytest.approx(1.0)


def test_metrics_partial_overlap():
    pred = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.float32)
    target = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.float32)

    assert measure.dice_overlap(pred, target) == pytest.approx(0.5, rel=1e-5)
    assert measure.intersection_over_union(pred, target) == pytest.approx(1.0 / 3.0, rel=1e-5)
    assert measure.accuracy(pred, target) == pytest.approx(0.5, rel=1e-5)
    assert measure.sensitivity(pred, target) == pytest.approx(0.5, rel=1e-5)
    assert measure.specificity(pred, target) == pytest.approx(0.5, rel=1e-5)


def test_metrics_average_over_batch_and_channels():
    pred = torch.tensor(
        [
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            [
                [[1, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [
                [[1, 0], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            [
                [[1, 0], [0, 1]],
                [[0, 0], [1, 1]],
            ],
        ],
        dtype=torch.float32,
    )

    per_case_dice = []
    per_case_iou = []
    per_case_acc = []
    per_case_sens = []
    per_case_spec = []

    for batch in range(pred.shape[0]):
        for channel in range(pred.shape[1]):
            p = pred[batch, channel]
            t = target[batch, channel]

            tp = torch.logical_and(p == 1, t == 1).sum().item()
            fp = torch.logical_and(p == 1, t == 0).sum().item()
            fn = torch.logical_and(p == 0, t == 1).sum().item()
            tn = torch.logical_and(p == 0, t == 0).sum().item()

            per_case_dice.append((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0)
            per_case_iou.append(tp / (tp + fp + fn) if (tp + fp + fn) else 1.0)
            per_case_acc.append((tp + tn) / (tp + tn + fp + fn))
            per_case_sens.append(tp / (tp + fn) if (tp + fn) else 1.0)
            per_case_spec.append(tn / (tn + fp) if (tn + fp) else 1.0)

    expected_dice = sum(per_case_dice) / len(per_case_dice)
    expected_iou = sum(per_case_iou) / len(per_case_iou)
    expected_acc = sum(per_case_acc) / len(per_case_acc)
    expected_sens = sum(per_case_sens) / len(per_case_sens)
    expected_spec = sum(per_case_spec) / len(per_case_spec)

    assert measure.dice_overlap(pred, target) == pytest.approx(expected_dice, rel=1e-5)
    assert measure.intersection_over_union(pred, target) == pytest.approx(expected_iou, rel=1e-5)
    assert measure.accuracy(pred, target) == pytest.approx(expected_acc, rel=1e-5)
    assert measure.sensitivity(pred, target) == pytest.approx(expected_sens, rel=1e-5)
    assert measure.specificity(pred, target) == pytest.approx(expected_spec, rel=1e-5)
