from train import train_temporal_model, load_flow_data
from temporal_model import TemporalStream
from utils import plot_and_save_loss_accuracy
import torch.optim as optim
import uuid


# --- Configuration ---
PLOTS_ROOT_DIR = f"plots"

if __name__ == "__main__":
    classes = 10
    channels = 18
    model = TemporalStream(num_classes=classes, num_channels=channels)

    trainset, valset, testset, train_loader, val_loader, test_loader= load_flow_data(batch_size=8, image_size=224)
    
    # --- Sanity check ---
    print(trainset.class_to_idx)
    print(valset.class_to_idx)  # must be identical

    y_tr = next(iter(train_loader))[1]
    y_va = next(iter(val_loader))[1]
    print("train labels sample:", y_tr[:10].tolist())
    print("val labels sample:",   y_va[:10].tolist())

    from collections import Counter

    train_labels = [label for _, label in trainset]
    val_labels = [label for _, label in valset]

    print("Train class distribution:", Counter(train_labels))
    print("Val class distribution:", Counter(val_labels))


    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    results = train_temporal_model(model, optimizer, train_loader, val_loader, trainset, valset,"cuda",500)

    plot_and_save_loss_accuracy(results, f"{PLOTS_ROOT_DIR}/temporal_figure_{uuid.uuid1()}")