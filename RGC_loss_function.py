import torch

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

    # Calculate the NPCC for the batch
    npcc_loss = npcc_loss.mean()

    return npcc_loss