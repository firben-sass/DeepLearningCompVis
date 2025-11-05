from train import train_optical_model, load_rgb_data
from optical_model import OpticalStream
from utils import plot_and_save_loss_accuracy
import torch.optim as optim
import uuid

# --- Configuration ---
PLOTS_ROOT_DIR = f"plots"

if __name__ == "__main__":
    classes = 10
    model = OpticalStream(num_classes=classes)
    trainset, valset, testset, train_loader, val_loader, test_loader= load_rgb_data(batch_size=8, image_size=224)

    # Train individual models
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    results = train_optical_model(model, optimizer, train_loader, val_loader, trainset, valset,"cuda",500)

    plot_and_save_loss_accuracy(results, f"{PLOTS_ROOT_DIR}/optical_figure_{uuid.uuid1()}")