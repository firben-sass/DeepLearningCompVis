import matplotlib.pyplot as plt

def plot_and_save_loss_accuracy(out_dict: dict, plots_path:str):
    # Create a figure with two subplots, one on top of the other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # Define the number of epochs from the length of one of the lists
    epochs = range(len(out_dict['train_acc']))

    # Plot 1: Accuracy
    ax1.plot(epochs, out_dict['train_acc'], 'o-', label='Train Accuracy')
    ax1.plot(epochs, out_dict['val_acc'], 'o-', label='Test Accuracy')
    ax1.set_title('Training & Test Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Loss
    ax2.plot(epochs, out_dict['train_loss'], 'o-', label='Train Loss')
    ax2.plot(epochs, out_dict['val_loss'], 'o-', label='Test Loss')
    ax2.set_title('Training & Test Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    ax2.grid(True)

    # --- Save Outputs ---
    output_png = f"{plots_path}.jpg"
    plt.savefig(output_png, bbox_inches="tight")