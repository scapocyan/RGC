import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from RGC_load_dataset import RGC_Dataset
from RGC_UNet_model import RGC_UNet
from tqdm import tqdm
from RGC_loss_function import NPCC_loss
from eval_metrics import PCC

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the training and validation directories
IMAGE_DIR = r'path\to\train\images'
LABEL_DIR = r'path\to\train\labels'

VAL_IMAGE_DIR = r'path\to\validation\images'
VAL_LABEL_DIR = r'path\to\validation\labels'

# Define the batch sizes
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2

# Load the dataset
training_set = RGC_Dataset(IMAGE_DIR, LABEL_DIR)
validation_set = RGC_Dataset(VAL_IMAGE_DIR, VAL_LABEL_DIR)

# Set the random seed before splitting the dataset
torch.manual_seed(0)

# Create the dataloaders
train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=VALID_BATCH_SIZE, pin_memory=True)

# Define the model and loss
model = RGC_UNet()
model = model.double()
model.to(device)

criterion = NPCC_loss

# Evaluating data to get the train/test accuracy and train/test loss
def evaluate(dataloader):
    # Set model to evaluate mode
    model.eval()
    
    loss_sum = 0
    pcc_sum = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(-1, 1, 512, 512)
            labels = labels.view(-1, 1, 512, 512)

            outputs = model(images)

            loss_sum += criterion(outputs, labels)
            pcc_sum += PCC(outputs, labels)

    # Calculate the training and validation accuracies and losses
    accuracy = pcc_sum.cpu() / len(dataloader)
    print_accuracy = accuracy * 100
    total_loss = loss_sum.cpu() / len(dataloader)

    if dataloader == train_dataloader:
        print(f"Train Accuracy: {print_accuracy:.2f}%, Train Loss: {total_loss:.5f}")
    elif dataloader == val_dataloader:
        print(f"Validation Accuracy: {print_accuracy:.2f}%, Validation Loss: {total_loss:.5f}\n")

    return accuracy, total_loss


# Set up the training loop
def train(model,
          optimizer,
          loss_fn,
          epochs=500,
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          **kwargs):
    
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n-------------------------------")
        
        # Set model to train mode
        model.train()
        
        # tqdm for progress bars
        loop = tqdm(train_dataloader)
        
        for batch, (images, labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(-1, 1, 512, 512)
            labels = labels.view(-1, 1, 512, 512)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate the model
        train_accuracy, train_loss = evaluate(train_dataloader)
        val_accuracy, val_loss = evaluate(val_dataloader)

        # Append the train and validation accuracies and losses to the lists
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
    
    # Plot the train and validation accuracies and losses
    plt.figure(1)
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Hyperparameters
EPOCHS = 50

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Train the model
train(model, optimizer, criterion, EPOCHS, train_dataloader, val_dataloader)