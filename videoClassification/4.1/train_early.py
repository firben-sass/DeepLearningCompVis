import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import FrameVideoDataset
from EarlyFusion import EarlyFusion
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

batch_size = 64
size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
val_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = FrameVideoDataset(
    split="train",
    transform=train_transform,
    root_dir="/zhome/1c/5/213743/Vision/project_2/ufc10"
)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, )
valset = FrameVideoDataset(
    split="val",
    transform=val_transform,
    root_dir="/zhome/1c/5/213743/Vision/project_2/ufc10"
)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, )

model = EarlyFusion(
    num_classes=10,
    p_dropout=0.25
)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use a lower learning rate for Adam
criterion = nn.CrossEntropyLoss()

train(model, 
    optimizer, 
    num_epochs=2, 
    save_name_model="early_fusion", 
    train_loader=train_loader, 
    val_loader=val_loader, 
    trainset=trainset, 
    valset=valset, 
    device=device)