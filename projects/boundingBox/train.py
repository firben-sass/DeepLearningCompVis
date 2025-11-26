import os
import sys
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


#We define the training as a function so we can easily re-use it.
def train(model, optimizer, num_epochs=10, save_name_model="model", train_loader=None, val_loader=None, trainset=None, valset=None, device='cuda'):
    # Define the loss function outside the loop
    criterion = nn.CrossEntropyLoss()

    out_dict = {'train_acc': [],
              'val_acc': [],
              'train_loss': [],
              'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = criterion(output, target) # Use the created instance
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Compute the val accuracy
        val_loss = []
        val_correct = 0
        model.eval()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            val_loss.append(criterion(output, target).cpu().item()) # Use the created instance
            predicted = output.argmax(1)
            val_correct += (target==predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['val_acc'].append(val_correct/len(valset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(np.mean(val_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t val: {np.mean(val_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t val: {out_dict['val_acc'][-1]*100:.1f}%")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.save(model.state_dict(), f"models/{save_name_model}.pth")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    print(out_dict)
    print(num_epochs)

    train_acc = list(map(float, out_dict['train_acc']))
    val_acc = list(map(float, out_dict['val_acc']))
    train_loss = list(map(float, out_dict['train_loss']))
    val_loss = list(map(float, out_dict['val_loss']))

    # --- Accuracy ---
    ax[0].plot(range(1, num_epochs + 1), train_acc, label="Train Accuracy")
    ax[0].plot(range(1, num_epochs + 1), val_acc, label="Validation Accuracy")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    # --- Loss ---
    ax[1].plot(range(1, num_epochs + 1), train_loss, label="Train Loss")
    ax[1].plot(range(1, num_epochs + 1), val_loss, label="Validation Loss")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"results/{save_name_model}_plot.png", bbox_inches='tight')
