import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optical_model import OpticalStream
from temporal_model import TemporalStream
from dual_model import DualStreamNetwork
from utils import plot_and_save_loss_accuracy
from datasets import FrameImageDataset, OpticalFlowDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
import uuid

# Config logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Config paths
DATASET_ROOT_DIR =  '/dtu/datasets1/02516/ucf101_noleakage'
SAVE_MODELS__DIR = 'pretrained_models'
# --- Configuration ---
PLOTS_ROOT_DIR = f"plots"

def create_models(classes=10):
    '''
    Checks models output shape
    Args:
        -classes = number of classes 
    '''
    optical_stream_model = None
    temporal_stream_model = None
    try:
        # Checking optical shape
        logging.info("Creating models...")
        optical_stream_model = OpticalStream(num_classes=classes)
        #rgb_sample = torch.randn(1, 3, 224, 224)
        #op_out = optical_stream_model(rgb_sample)
        logging.info(f"Created optical model.")
        
        # Checking temporal shape
        temporal_stream_model = TemporalStream(num_classes=classes, num_channels=18)
        #flow_sample = torch.randn(1, 18, 224, 224)
        #temp_out = temporal_stream_model(flow_sample)
        logging.info(f"Created temporal model")
    except Exception as e:
        logging.error(f"Error checking model: {e}")
    return optical_stream_model, temporal_stream_model

def load_rgb_data(batch_size=64, image_size=224):
    logging.info("Loading RGB dataset")
    transform = T.Compose([T.Resize((image_size, image_size)),T.ToTensor()])
    # Train 
    trainset = FrameImageDataset(root_dir=DATASET_ROOT_DIR, split='train', transform=transform)
    train_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=True, num_workers=4)
    # Val
    valset = FrameImageDataset(root_dir=DATASET_ROOT_DIR, split='val', transform=transform)
    val_loader = DataLoader(valset,  batch_size=batch_size, shuffle=False, num_workers=4)
    # Test
    testset = FrameImageDataset(root_dir=DATASET_ROOT_DIR, split='test', transform=transform)
    test_loader = DataLoader(testset,  batch_size=batch_size, shuffle=False)
    return trainset, valset, testset, train_loader, val_loader, test_loader

def load_flow_data(batch_size=64, image_size=224):
    logging.info("Loading flow dataset")
    trainset = OpticalFlowDataset(root_dir=DATASET_ROOT_DIR, split='train', image_size=image_size)
    valset = OpticalFlowDataset(root_dir=DATASET_ROOT_DIR, split='val', image_size=image_size)
    testset = OpticalFlowDataset(root_dir=DATASET_ROOT_DIR, split='test')
    
    train_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset,  batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(testset,  batch_size=batch_size, shuffle=False)
    return trainset, valset, testset, train_loader, val_loader, test_loader

def train_optical_model(model:OpticalStream, optimizer, train_loader, val_loader, trainset, valset,
                     device, num_epochs=10):
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    out_dict = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'val_loss': []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_loss = []

        # --- Training loop ---
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)  # Forward pass dual stream

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()

        # --- Validation loop ---
        model.eval()
        val_loss = []
        val_correct = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            val_loss.append(criterion(output, target).cpu().item())
            predicted = output.argmax(1)
            val_correct += (target == predicted).sum().cpu().item()

        # --- Logging ---
        train_acc = train_correct / len(trainset)
        val_acc = val_correct / len(valset)
        out_dict['train_acc'].append(train_acc)
        out_dict['test_acc'].append(val_acc)
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(np.mean(val_loss))

        print(f"Epoch {epoch+1}: Loss train {np.mean(train_loss):.3f}, test {np.mean(val_loss):.3f} | "
              f"Accuracy train {out_dict['train_acc'][-1]*100:.1f}%, test {out_dict['test_acc'][-1]*100:.1f}%")
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{SAVE_MODELS__DIR}/optical_best.pt")
            print(f"Saved new best model with val_acc={val_acc*100:.2f}%")
    return out_dict
def train_temporal_model(model:TemporalStream, 
                        optimizer, 
                        train_loader,
                        val_loader, trainset, 
                        valset,
                        device, 
                        num_epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    out_dict = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'val_loss': []}
    best_val_acc = 0.0
    logging.info("Training temporal model")
    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_loss = []

        # --- Training loop ---
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)  # Forward pass dual stream

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()

        # --- Validation loop ---
        model.eval()
        val_loss = []
        val_correct = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            val_loss.append(criterion(output, target).cpu().item())
            predicted = output.argmax(1)
            val_correct += (target == predicted).sum().cpu().item()

        # --- Logging ---
        train_acc = train_correct / len(trainset)
        val_acc = val_correct / len(valset)
        out_dict['train_acc'].append(train_acc)
        out_dict['test_acc'].append(val_acc)
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(np.mean(val_loss))

        print(f"Epoch {epoch+1}: Loss train {np.mean(train_loss):.3f}, test {np.mean(val_loss):.3f} | "
              f"Accuracy train {out_dict['train_acc'][-1]*100:.1f}%, test {out_dict['test_acc'][-1]*100:.1f}%")
        
                # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{SAVE_MODELS__DIR}/temporal_best.pt")
            print(f"Saved new best model with val_acc={val_acc*100:.2f}%")

    return out_dict
    
def main():
    # Create models
    optical_model, temporal_model = create_models(classes=10)
    frame_trainset, frame_valset, frame_testset, train_frame_loader, val_frame_loader, test_frame_loader= load_rgb_data(batch_size=8, image_size=224)
    flow_trainset, flow_valset, flow_testset, train_flow_loader, val_flow_loader, test_flow_loader= load_flow_data(batch_size=64, image_size=224)
    
    optimizer = optim.Adam(optical_model.parameters(), lr=0.0001) 
    optical_results = train_optical_model(optical_model, optimizer, train_frame_loader, val_frame_loader, frame_trainset, frame_valset,"cuda",10)
    temporal_results = train_temporal_model(temporal_model, optimizer, train_flow_loader, val_flow_loader, flow_trainset, flow_valset,"cuda",10)

    plot_and_save_loss_accuracy(optical_results, f"{PLOTS_ROOT_DIR}/optical_figure_{uuid.uuid1()}")
    plot_and_save_loss_accuracy(temporal_results, f"{PLOTS_ROOT_DIR}/temporal_figure_{uuid.uuid1()}")
            

if __name__ == "__main__":
    main() 