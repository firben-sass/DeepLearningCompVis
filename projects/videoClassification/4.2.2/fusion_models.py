import torch
import torch.nn as nn
from optical_model import OpticalStream
from temporal_model import TemporalStream  # or whatever your flow model class is
from fusion_model import TwoStreamFusion
from fusion_dataset import RGBFlowPairDataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10

# --- Load pretrained single streams ---
rgb_model = OpticalStream(num_classes=NUM_CLASSES).to(DEVICE) #Using RGB Resnet
rgb_model.load_state_dict(torch.load("pretrained_models/optical_normal_best.pt"))
rgb_model.eval()

flow_model = TemporalStream(num_classes=NUM_CLASSES, num_channels=18).to(DEVICE)#Own flow model
flow_model.load_state_dict(torch.load("pretrained_models/temporal_best_2.pt"))
flow_model.eval()

# --- Fusion model ---
fusion_model = TwoStreamFusion(num_classes=NUM_CLASSES).to(DEVICE)

# --- Dataset for evaluation ---
dataset = RGBFlowPairDataset("/dtu/datasets1/02516/ucf101_noleakage", split="val", image_size=224, n_frames=10, aug_rgb=False, aug_flow=False)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

@torch.no_grad()
def evaluate_fusion(alpha=None):
    fusion_model.eval()
    total, correct = 0, 0
    for rgb, flow, y in loader:
        rgb, flow, y = rgb.to(DEVICE), flow.to(DEVICE), y.to(DEVICE)
        logits_rgb = rgb_model(rgb)
        logits_flow = flow_model(flow)
        if alpha is not None:
            logits = alpha * logits_rgb + (1 - alpha) * logits_flow
        else:
            logits = fusion_model(logits_rgb, logits_flow)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

# --- Option 1a: fixed weight fusion ---
#for a in [0.3, 0.5, 0.7]:
#    acc = evaluate_fusion(alpha=a)
#    print(f"Fixed α={a:.1f}: accuracy={acc*100:.2f}%")

# --- Option 1b: learnable α (fine-tune) ---
optimizer = torch.optim.Adam([fusion_model.alpha], lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    total_loss = 0.0
    for rgb, flow, y in loader:
        rgb, flow, y = rgb.to(DEVICE), flow.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            logits_rgb = rgb_model(rgb)
            logits_flow = flow_model(flow)
        logits = fusion_model(logits_rgb, logits_flow)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d} fusion loss={total_loss/len(loader):.4f}, α={fusion_model.alpha.item():.3f}")

acc = evaluate_fusion()
print(f"Final fusion accuracy={acc*100:.2f}%  with learned α={fusion_model.alpha.item():.3f}")
