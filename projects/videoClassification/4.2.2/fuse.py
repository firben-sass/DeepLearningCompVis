import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_fusion(
    spatial_model, temporal_model,
    spatial_loader, temporal_loader,
    device,
    weight_spatial=0.5,
    weight_temporal=0.5,
    fuse_after_softmax=True
):
    """
    Fuse predictions from spatial (RGB) and temporal (Flow) models.

    Args:
        spatial_model: trained optical (RGB) model
        temporal_model: trained temporal (flow) model
        spatial_loader: dataloader for RGB test/val data
        temporal_loader: dataloader for flow test/val data
        device: torch device
        weight_spatial: fusion weight for RGB stream
        weight_temporal: fusion weight for Flow stream
        fuse_after_softmax: if True, fuse probabilities; else fuse raw logits

    Returns:
        Fused accuracy (float)
    """
    spatial_model.eval()
    temporal_model.eval()
    spatial_model.to(device)
    temporal_model.to(device)

    total = 0
    correct = 0

    with torch.no_grad():
        for (rgb, target), (flow, _) in tqdm(zip(spatial_loader, temporal_loader), total=len(spatial_loader)):
            rgb, flow, target = rgb.to(device), flow.to(device), target.to(device)

            # Forward passes
            rgb_out = spatial_model(rgb)
            flow_out = temporal_model(flow)

            # Normalize or not depending on fusion type
            if fuse_after_softmax:
                rgb_out = F.softmax(rgb_out, dim=1)
                flow_out = F.softmax(flow_out, dim=1)

            # Weighted fusion
            fused_out = weight_spatial * rgb_out + weight_temporal * flow_out

            preds = fused_out.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    acc = correct / total
    print(f"\nFused Accuracy: {acc * 100:.2f}% (weights {weight_spatial}:{weight_temporal})")
    return acc
