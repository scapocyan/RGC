import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import stats as st

class RGC_Dataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """This function should return an image and its label given an index. The images and labels are 16-bit grayscale images.
        The images and labels are both of size 1040x1392. The images are stored in the image_dir and the labels are stored in the label_dir.

        Args:
            index (int): The index of the image and label to be returned.
        """
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.labels[index])
        image = Image.open(img_path)
        label = Image.open(label_path)

        # Convert to numpy array before normalization
        image = np.array(image)
        label = np.array(label)

        # Image Normalization
        image = image.astype(np.double) # convert to float before normalization
        image_mode = st.mode(image, axis=None, keepdims=False)
        image -= image_mode[0]
        image[image < 0] = 0
        image /= np.max(image)
        image = torch.from_numpy(image)

        # Label normalization
        label = label.astype(np.double)
        label_mode = st.mode(label, axis=None, keepdims=False)
        label -= label_mode[0]
        label[label < 0] = 0
        label /= np.max(label)
        label = torch.from_numpy(label)

        return image, label

# # Load the dataset
# image_dir = r'C:\Users\Sam\Documents\Python Scripts\RGC\Datasets\Cropped_03_maxdiv7\train_images'
# label_dir = r'C:\Users\Sam\Documents\Python Scripts\RGC\Datasets\Cropped_03_maxdiv7\train_labels'
# training_images = RGC_Dataset(image_dir, label_dir, transform=None)

# torch.manual_seed(0)

# # Split the dataset into training and validation sets
# train_size = int(0.9 * len(training_images))
# val_size = len(training_images) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(training_images, [train_size, val_size])

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# # Print the shape of the inputs in the dataloaders
# for image, label in train_dataloader:
#     image.unsqueeze_(1)
#     label.unsqueeze_(1)
#     print(image.shape)
#     print(label.shape)
#     break

# # Check the data type of the images and labels
# for image, label in train_dataloader:
#     print(image.dtype)
#     print(label.dtype)
#     break

# # # Check the shape of the images and labels
# # counter = 0
# # for image, label in train_dataloader:
# #     print(image.shape)
# #     print(label.shape)
# #     if counter == 2:
# #         break
# #     counter += 1

# # Display a random sample image and label
# image, label = train_dataset[0]
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Image')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(label, cmap='gray')
# plt.title('Label')
# plt.axis('off')
# plt.show()

# # Print the size of the training and validation sets
# print(f"Training set size: {len(train_dataset)}")
# print(f"Validation set size: {len(val_dataset)}")

# # Loop through training_dataset until you find images/labels with 'nan' values, then print the index of the image/label
# for i in range(len(train_dataset)):
#     image, label = train_dataset[i]
#     if torch.isnan(image).any() or torch.isnan(label).any():
#         print(i)
#         break

# # Count how many images/labels in the training_dataset have 'nan' values
# counter = 0
# for i in range(len(train_dataset)):
#     image, label = train_dataset[i]
#     if torch.isnan(image).any() or torch.isnan(label).any():
#         counter += 1
# print(f"Number of images/labels with 'nan' values: {counter}")

# # Calculate the mode and max pixel values of a specific image
# image, label = train_dataset[1337]
# image_elements, image_counts = torch.unique(image.view(-1), return_counts=True)
# image_mode_index = torch.argmax(image_counts)
# image_mode = image_elements[image_mode_index]
# image_max = torch.max(image)
# print(f"Image mode: {image_mode}")
# print(f"Image max: {image_max}")