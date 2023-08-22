import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from RGC_dataloader import RGC_Dataset
from RGC_UNet_model import RGC_UNet
from tqdm import tqdm

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the directory paths
IMAGE_DIR = r'C:\Users\Sam\Documents\Python Scripts\RGC\RGC_unadjusted_dataset\train_images'
LABEL_DIR = r'C:\Users\Sam\Documents\Python Scripts\RGC\RGC_unadjusted_dataset\train_labels'

# Define the batch sizes
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8

# Load the dataset
training_set = RGC_Dataset(IMAGE_DIR, LABEL_DIR, transform=None)

train_size = int(0.9 * len(training_set))
val_size = len(training_set) - train_size

# Set the random seed before splitting the dataset
torch.manual_seed(0)

train_dataset, val_dataset = torch.utils.data.random_split(training_set, [train_size, val_size])

# Create the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True, pin_memory=True)


# Define Negative Pearson Correlation Coefficient loss
def NPCC_loss(outputs, labels):
    # Calculate the mean of the outputs and labels
    outputs_mean = outputs.mean(dim=(2,3), keepdim=True)
    labels_mean = labels.mean(dim=(2,3), keepdim=True)

    # Subract the mean from the outputs and labels
    outputs = outputs - outputs_mean
    labels = labels - labels_mean

    # Multiply the outputs and labels
    outputs_labels = torch.mul(outputs, labels)

    # Square the outputs and labels
    outputs_squared = torch.square(outputs)
    labels_squared = torch.square(labels)

    # Calculate the sum of the elements in the outputs_labels, outputs_squared and labels_squared tensors
    outputs_labels_sum = torch.sum(outputs_labels, dim=(2,3))
    outputs_squared_sum = torch.sum(outputs_squared, dim=(2,3))
    labels_squared_sum = torch.sum(labels_squared, dim=(2,3))

    # Calculate the square root of outputs_squared_sum and labels_squared_sum
    outputs_squared_sum_sqrt = torch.sqrt(outputs_squared_sum)
    labels_squared_sum_sqrt = torch.sqrt(labels_squared_sum)

    # Calculate the numerator and denominator
    numerator = -1 * outputs_labels_sum
    denominator = outputs_squared_sum_sqrt * labels_squared_sum_sqrt

    # Calculate the NPCC
    npcc_loss = torch.div(numerator, denominator)

    # # Calculate the NPCC for the batch
    # npcc_loss = npcc_loss.mean()

    return npcc_loss


# Define PCC as for evaluating performance
def PCC(outputs, labels):
    # Calculate the mean of the outputs and labels
    outputs_mean = outputs.mean(dim=(2,3), keepdim=True)
    labels_mean = labels.mean(dim=(2,3), keepdim=True)

    # Subract the mean from the outputs and labels
    outputs = outputs - outputs_mean
    labels = labels - labels_mean

    # Multiply the outputs and labels
    outputs_labels = torch.mul(outputs, labels)

    # Square the outputs and labels
    outputs_squared = torch.square(outputs)
    labels_squared = torch.square(labels)

    # Calculate the sum of the elements in the outputs_labels, outputs_squared and labels_squared tensors
    outputs_labels_sum = torch.sum(outputs_labels, dim=(2,3))
    outputs_squared_sum = torch.sum(outputs_squared, dim=(2,3))
    labels_squared_sum = torch.sum(labels_squared, dim=(2,3))

    # Calculate the square root of outputs_squared_sum and labels_squared_sum
    outputs_squared_sum_sqrt = torch.sqrt(outputs_squared_sum)
    labels_squared_sum_sqrt = torch.sqrt(labels_squared_sum)

    # Calculate the numerator and denominator
    numerator = outputs_labels_sum
    denominator = outputs_squared_sum_sqrt * labels_squared_sum_sqrt + 1e-10

    # Calculate the PCC
    pcc = torch.div(numerator, denominator)

    # # Calculate the PCC for the batch
    # pcc = pcc.mean()

    return pcc


# Define the model and loss
model = RGC_UNet()
model = model.double()
model.to(device)
criterion = NPCC_loss

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Evaluating data to get the train/test accuracy and train/test loss
def evaluate(dataloader):
    loss_sum = 0
    pcc_sum = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(-1, 1, 256, 256)
            labels = labels.view(-1, 1, 256, 256)

            outputs = model(images)

            loss_sum += criterion(outputs, labels).sum()
            pcc_sum += PCC(outputs, labels).sum()

    # Calculate the training and validation accuracies and losses
    accuracy = pcc_sum.cpu() / len(dataloader.dataset)
    total_loss = loss_sum.cpu() / len(dataloader.dataset)

    if dataloader == train_dataloader:
        print(f"Train Accuracy: {accuracy:.2f}%, Train Loss: {total_loss:.5f}")
    elif dataloader == val_dataloader:
        print(f"Validation Accuracy: {accuracy:.2f}%, Validation Loss: {total_loss:.5f}")

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

        loop = tqdm(train_dataloader)

        for batch, (images, labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(-1, 1, 256, 256)
            labels = labels.view(-1, 1, 256, 256)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss = torch.mean(loss)

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
    plt.title('Train and Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Train and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Hyperparameters
EPOCHS = 3

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Train the model
train(model, optimizer, criterion, EPOCHS, train_dataloader, val_dataloader)

