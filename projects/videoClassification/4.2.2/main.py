import os
import numpy as np
import torch
import torch.nn as nn
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
DATASET_ROOT_DIR =  '/home/datawa/Repositories/ucf101_noleakage'
# --- Configuration ---
plotspath = f"plots/figure_{uuid.uuid1()}"

def create_models(classes=10):
    '''
    Checks models output shape
    Args:
        -classes = number of classes 
    '''
    optical_stream_model = None
    temporal_stream_model = None
    dual_stream_model = None
    try:
        # Checking optical shape
        logging.info("Creating models...")
        optical_stream_model = OpticalStream(num_classes=classes)
        rgb_sample = torch.randn(1, 3, 224, 224)
        op_out = optical_stream_model(rgb_sample)
        logging.info(f"Created optical model. Fast check with input {rgb_sample.shape} = {op_out.shape})")
        
        # Checking temporal shape
        temporal_stream_model = TemporalStream(num_classes=classes, num_channels=9)
        flow_sample = torch.randn(1, 9, 224, 224)
        temp_out = temporal_stream_model(flow_sample)
        logging.info(f"Created temporal model. Fast check with input {flow_sample.shape} = {temp_out.shape})")

        # Checking dual stream shape
        dual_stream_model = DualStreamNetwork(num_classes=classes)
        dual_out = dual_stream_model(rgb_sample, flow_sample)
        logging.info(f"Created dual stream network. Fast check with input {rgb_sample.shape} and {flow_sample.shape}  = {dual_out.shape})")
    except Exception as e:
        logging.error(f"Error chechking models dual stream model: {e}")
    return optical_stream_model, temporal_stream_model, dual_stream_model

def load_frame_dataset(batch_size=64, image_size=224):
    logging.info("Loading RGB dataset")
    transform = T.Compose([T.Resize((image_size, image_size)),T.ToTensor()])
    # Train 
    trainset = FrameImageDataset(root_dir=DATASET_ROOT_DIR, split='train', transform=transform)
    train_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=False)
    # Val
    valset = FrameImageDataset(root_dir=DATASET_ROOT_DIR, split='val', transform=transform)
    val_loader = DataLoader(valset,  batch_size=batch_size, shuffle=False)
    # Test
    testset = FrameImageDataset(root_dir=DATASET_ROOT_DIR, split='test', transform=transform)
    test_loader = DataLoader(testset,  batch_size=batch_size, shuffle=False)
    return trainset, valset, testset, train_loader, val_loader, test_loader

def load_flow_dataset(batch_size=16, image_size=224):
    logging.info("Loading flow dataset")
    transform = T.Compose([T.Resize((image_size, image_size)),T.ToTensor()])
    # Train 
    trainset = OpticalFlowDataset(root_dir=DATASET_ROOT_DIR, split='train', transform=transform)
    train_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=False)
    # Val
    valset = OpticalFlowDataset(root_dir=DATASET_ROOT_DIR, split='val', transform=transform)
    val_loader = DataLoader(valset,  batch_size=batch_size, shuffle=False)
    # Test
    testset = OpticalFlowDataset(root_dir=DATASET_ROOT_DIR, split='test', transform=transform)
    test_loader = DataLoader(testset,  batch_size=batch_size, shuffle=False)
    return trainset, valset, testset, train_loader, val_loader, test_loader

def train_optical_model(model:OpticalStream, optimizer, train_loader, val_loader, trainset, valset,
                     device, num_epochs=10):
    
    def loss_fun(output, target):
        return F.nll_loss(torch.log(output), target) 

    out_dict = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_loss = []

        # --- Training loop ---
        for _, data, labels in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data)  # Forward pass dual stream
            loss = loss_fun(output, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            predicted = output.argmax(1)
            train_correct += (predicted == labels).sum().cpu().item()

        # --- Validation loop ---
        model.eval()
        test_loss = []
        test_correct = 0
        for _, data, labels in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, labels).cpu().item())
            predicted = output.argmax(1)
            test_correct += (predicted == labels).sum().cpu().item()

        # --- Logging ---
        out_dict['train_acc'].append(train_correct / len(trainset))
        out_dict['test_acc'].append(test_correct / len(valset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))

        print(f"Epoch {epoch+1}: Loss train {np.mean(train_loss):.3f}, test {np.mean(test_loss):.3f} | "
              f"Accuracy train {out_dict['train_acc'][-1]*100:.1f}%, test {out_dict['test_acc'][-1]*100:.1f}%")

    return out_dict
#def train_temporal_model():

def train_dualstream(model:DualStreamNetwork, optimizer, train_rgb_loader, train_flow_loader,
                     val_rgb_loader, val_flow_loader, trainset, valset,
                     device, num_epochs=10):
    
    def loss_fun(output, target):
        return F.nll_loss(torch.log(output), target) 

    out_dict = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_loss = []

        # --- Training loop ---
        for _, ((rgb_data, labels), (flow_data, _)) in enumerate(zip(train_rgb_loader, train_flow_loader)):
            rgb_data, flow_data, labels = rgb_data.to(device), flow_data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(rgb_data, flow_data)  # Forward pass dual stream
            loss = loss_fun(output, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            predicted = output.argmax(1)
            train_correct += (predicted == labels).sum().cpu().item()

        # --- Validation loop ---
        model.eval()
        test_loss = []
        test_correct = 0
        for (rgb_data, labels), (flow_data, _) in zip(val_rgb_loader, val_flow_loader):
            rgb_data, flow_data, labels = rgb_data.to(device), flow_data.to(device), labels.to(device)
            with torch.no_grad():
                output = model(rgb_data, flow_data)
            test_loss.append(loss_fun(output, labels).cpu().item())
            predicted = output.argmax(1)
            test_correct += (predicted == labels).sum().cpu().item()

        # --- Logging ---
        out_dict['train_acc'].append(train_correct / len(trainset))
        out_dict['test_acc'].append(test_correct / len(valset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))

        print(f"Epoch {epoch+1}: Loss train {np.mean(train_loss):.3f}, test {np.mean(test_loss):.3f} | "
              f"Accuracy train {out_dict['train_acc'][-1]*100:.1f}%, test {out_dict['test_acc'][-1]*100:.1f}%")

    return out_dict

def main():
    optical_model, _ , _ = create_models()
    frame_trainset, frame_valset, frame_testset, train_frame_loader, val_frame_loader, test_frame_loader= load_frame_dataset(batch_size=64, image_size=224)
    flow_trainset, flow_valset, flow_testset, train_flow_loader, val_flow_loader, test_flow_loader= load_flow_dataset(batch_size=64, image_size=224)
    # --- Chech dataset sizes ---
    logging.info(f"Frame train dataset size: {len(frame_trainset)}")
    for video_frames, labels in train_frame_loader:
        logging.info(f"{video_frames.shape}{labels.shape}") # [batch, channels, height, width]
    
    #logging.info(f"Flow train dataset size: {len(flow_trainset)}")
    #for flow, labels in train_flow_loader:
    #    logging.info(f"{flow.shape}{labels.shape}") # [batch, channels, height, width]
    

    optimizer = torch.optim.Adam(optical_model.parameters(), lr=1e-4)
    optical_results = train_optical_model(optical_model, optimizer, train_frame_loader, val_frame_loader, frame_trainset, frame_valset,"cuda",10)
    #results = train_dualstream(dual_stream_model, optimizer, train_frame_loader, train_flow_loader, val_frame_loader, val_flow_loader, frame_trainset, frame_valset, "gpu",5)

    #plot_and_save_loss_accuracy(results, plotspath)
            

if __name__ == "__main__":
    main() 