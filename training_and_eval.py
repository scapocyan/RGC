import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Evaluate the model
def evaluate(model,
             loss_fn,
             eval_metric,
             dataloader,
             **kwargs):
    # Set model to evaluate mode
    model.eval()
    
    loss_sum = 0
    pcc_sum = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(kwargs['device'])
            labels = labels.to(kwargs['device'])

            # Forward pass
            outputs = model(images)

            loss_sum += loss_fn(outputs, labels)
            pcc_sum += eval_metric(outputs, labels)

    # Calculate the training and validation accuracies and losses
    accuracy = pcc_sum.cpu() / len(dataloader)
    print_accuracy = accuracy * 100
    total_loss = loss_sum.cpu() / len(dataloader)

    if kwargs['dataloader_type'] == 'train':
        print(f"Training Accuracy: {print_accuracy:.2f}%, Training Loss: {total_loss:.5f}")
    elif kwargs['dataloader_type'] == 'validation':
        print(f"Validation Accuracy: {print_accuracy:.2f}%, Validation Loss: {total_loss:.5f}\n")

    return accuracy, total_loss


# Set up the training loop
def train(model,
          optimizer,
          loss_fn,
          epochs=500,
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
        loop = tqdm(kwargs['train_dataloader'])
        
        for _, (images, labels) in enumerate(loop):
            images = images.to(kwargs['device'])
            labels = labels.to(kwargs['device'])

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model
        train_accuracy, train_loss = evaluate(model, loss_fn, kwargs['eval_metric'],
                                              kwargs['train_dataloader'],
                                              dataloader_type='train',
                                              device=kwargs['device'])
        val_accuracy, val_loss = evaluate(model, loss_fn, kwargs['eval_metric'], 
                                          kwargs['val_dataloader'],
                                          dataloader_type='validation',
                                          device=kwargs['device'])

        # Append the accuracies and losses to the lists
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

    # Plot the accuracies and losses
    plt.figure(1)
    plt.plot(train_accuracies, label='Training')
    plt.plot(val_accuracies, label='Validation')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(train_losses, label='Training')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()