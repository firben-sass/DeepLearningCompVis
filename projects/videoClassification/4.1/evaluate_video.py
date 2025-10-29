import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from datasets import FrameVideoDataset
from EarlyFusion import EarlyFusion
from LateFusion import LateFusion
from Simple3DCNN import Simple3DCNN


def evaluate_video_model(model,test_loader, device, name="Model"):
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc='Testing'):
            videos = videos.to(device)  # (batch, 3, T, H, W)
            
            outputs = model(videos)  # (batch, num_classes)
            predictions = outputs.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    accuracy = accuracy_score(labels, predictions)

    print(f"Test Accuracy of {name}: {accuracy * 100:.2f}%")


batch_size = 64
size = 128
test_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
test_dataset = FrameVideoDataset(
    root_dir="/zhome/1c/5/213743/Vision/project_2/ufc10",
    split='test',
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size,  # Smaller batch size for video models
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = EarlyFusion(
    num_classes=10,
    p_dropout=0.25
)
model.load_state_dict(torch.load('early_fusion.pth'))

evaluate_video_model(model, test_loader, device, name="early_fusion")

model = LateFusion(
    num_classes=10,
    p_dropout=0.25
)
model.load_state_dict(torch.load('late_fusion.pth'))

evaluate_video_model(model, test_loader, device, name="late_fusion")

model = Simple3DCNN(
    num_classes=10,
    p_dropout=0.25
)
model.load_state_dict(torch.load('Simple3DCNN.pth'))

evaluate_video_model(model, test_loader, device, name="Simple3DCNN")

test_dataset = FrameVideoDataset(
    root_dir="/dtu/datasets1/02516/ucf101_noleakage",
    split='test',
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size,  # Smaller batch size for video models
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

model = LateFusion(
    num_classes=10,
    p_dropout=0.25
)
model.load_state_dict(torch.load('late_fusion_noleak.pth'))

evaluate_video_model(model, test_loader, device, name="late_fusion_noleak")

model = Simple3DCNN(
    num_classes=10,
    p_dropout=0.25
)
model.load_state_dict(torch.load('Simple3DCNN_noleak.pth'))

evaluate_video_model(model, test_loader, device, name="Simple3DCNN_noleak")