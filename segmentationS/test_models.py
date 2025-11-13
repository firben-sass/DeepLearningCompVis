#!/usr/bin/env python3
"""
Test script to demonstrate the different models and loss functions implemented for Project 3
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary

# Import our models and losses
from lib.model.EncDecModel import EncDec
from lib.model.DilatedNetModel import DilatedNet
from lib.model.UNetModel import UNet, UNet2
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation
from lib.dataset.PhCDataset import PhC

def test_models():
    """Test all implemented models with a sample input"""
    print("Testing Models...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample input (batch_size=2, channels=3, height=128, width=128)
    sample_input = torch.randn(2, 3, 128, 128).to(device)
    
    models = {
        'EncDecModel': EncDec(),
        'DilatedNet': DilatedNet(),
        'UNet': UNet(n_channels=3, n_classes=1),
        'UNet2': UNet2(n_channels=3, n_classes=1)
    }
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        model = model.to(device)
        model.eval()
        
        try:
            with torch.no_grad():
                output = model(sample_input)
            print(f"  Input shape: {sample_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # Print model summary for first model only to avoid clutter
            if model_name == 'EncDecModel':
                print("  Model summary:")
                summary(model, (3, 128, 128))
                
        except Exception as e:
            print(f"  Error: {e}")
    
def test_loss_functions():
    """Test all implemented loss functions"""
    print("\n\nTesting Loss Functions...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample predictions and targets
    batch_size, channels, height, width = 2, 1, 64, 64
    y_pred = torch.randn(batch_size, channels, height, width).to(device)  # Logits
    y_true = torch.randint(0, 2, (batch_size, channels, height, width)).float().to(device)  # Binary mask
    
    loss_functions = {
        'BCELoss': BCELoss(),
        'DiceLoss': DiceLoss(),
        'FocalLoss': FocalLoss(),
        'BCELoss_TotalVariation': BCELoss_TotalVariation()
    }
    
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Target shape: {y_true.shape}")
    print(f"Target range: [{y_true.min().item():.1f}, {y_true.max().item():.1f}]")
    
    for loss_name, loss_fn in loss_functions.items():
        try:
            loss_value = loss_fn(y_pred, y_true)
            print(f"\n{loss_name}: {loss_value.item():.6f}")
        except Exception as e:
            print(f"\n{loss_name}: Error - {e}")

def test_dataset():
    """Test the dataset loading"""
    print("\n\nTesting Dataset...")
    print("="*50)
    
    try:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Test train set
        trainset = PhC(train=True, transform=transform)
        print(f"Training images: {len(trainset)}")
        
        if len(trainset) > 0:
            sample_image, sample_label = trainset[0]
            print(f"Sample image shape: {sample_image.shape}")
            print(f"Sample label shape: {sample_label.shape}")
            print(f"Image range: [{sample_image.min():.4f}, {sample_image.max():.4f}]")
            print(f"Label range: [{sample_label.min():.4f}, {sample_label.max():.4f}]")
        
        # Test data loader
        train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
        batch_images, batch_labels = next(iter(train_loader))
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        
    except Exception as e:
        print(f"Dataset error: {e}")

def train_sample():
    """Run a short training sample with different models and losses"""
    print("\n\nRunning Sample Training...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    trainset = PhC(train=True, transform=transform)
    if len(trainset) == 0:
        print("No training data available!")
        return
        
    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    
    # Test combinations
    combinations = [
        ('UNet', UNet(n_channels=3, n_classes=1), 'DiceLoss', DiceLoss()),
        ('UNet2', UNet2(n_channels=3, n_classes=1), 'FocalLoss', FocalLoss()),
        ('DilatedNet', DilatedNet(n_channels=3, n_classes=1), 'BCELoss', BCELoss()),
    ]
    
    for model_name, model, loss_name, loss_fn in combinations:
        print(f"\nTesting {model_name} + {loss_name}")
        
        model = model.to(device)
        model.train()
        
        # Get one batch
        try:
            X_batch, y_true = next(iter(train_loader))
            X_batch = X_batch.to(device)
            y_true = y_true.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_true)
            
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Predictions shape: {y_pred.shape}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    print("Project 3 - Image Segmentation Implementation Test")
    print("="*60)
    
    test_models()
    test_loss_functions() 
    test_dataset()
    train_sample()
    
    print("\n" + "="*60)
    print("Testing completed!")
