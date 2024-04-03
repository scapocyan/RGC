import torch
import torch.optim as optim
from RGC_load_dataset import single_phase_dataset
from models import single_phase_UNet
from RGC_loss_function import NPCC_loss
from eval_metrics import PCC
from training_and_eval import train


# Main function
def main(train_batch: int,
         valid_batch: int,
         epochs: int = 500):
    """Main function to train the model.

    Args:
        train_batch (int): Defines the batch size for training.
        valid_batch (int): Defines the batch size for validation.
        epochs (int, optional): Defines the number of epochs for training. Defaults to 500.
    """

    # Set the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

   # Load the dataset
    training_set = single_phase_dataset(IMAGE_DIR, LABEL_DIR)

    # Define the training/validation split
    train_size = int(0.9 * len(training_set))
    val_size = len(training_set) - train_size

    # Set the random seed before splitting the dataset
    torch.manual_seed(0)

    # Split the dataset into training and validation sets
    training_set, validation_set = torch.utils.data.random_split(training_set, [train_size, val_size])

    # Create the dataloaders
    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=train_batch, shuffle=True, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=valid_batch, pin_memory=True)

    # Define the model
    model = single_phase_UNet()
    model.to(device)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Train the model
    train(model,
          optimizer,
          NPCC_loss,
          epochs,
          eval_metric=PCC,
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          device=device)
    

# Define the training directories
IMAGE_DIR = r'path\to\train\images'
LABEL_DIR = r'path\to\train\labels'
    

# Run the main function
if __name__ == "__main__":
    main(8, 2, 50)

