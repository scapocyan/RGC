import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


# Dataset class for the single phase dataset
class single_phase_dataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """This function should return an image input stack and its label given an index.
        The image input stacks and labels are 16-bit grayscale images. The images and labels are
        already cropped with the size of 512x512. The images are stored in the image_dir,
        and the labels are stored in the label_dir.

        Args:
            index (int): The index of the image input stack and label to be returned.
        """
        # Extract the images from the subdirectories
        subdir = self.images[index]
        subdir_path = os.path.join(self.image_dir, subdir)
        
        image_names = ['top.tif', 'bottom.tif', 'left.tif', 'right.tif', 'dpc_tb.tif', 'dpc_lr.tif']
        images = [Image.open(os.path.join(subdir_path, image)) for image in image_names]
        images = [np.array(image) for image in images]
        images = [image.astype(np.float32) / 65535 for image in images]
        images = np.stack(images, axis=-1)

        # Extract the label from the subdirectory
        label_path = os.path.join(self.label_dir, self.labels[index])
        label = Image.open(label_path)
        label = np.array(label)
        label = label.astype(np.float32) / 65535

        # Convert to torch tensors
        images = torch.from_numpy(images)
        images = images.permute(2, 0, 1) # Change the shape from (H, W, C) to (C, H, W)

        label = label[np.newaxis] # Add a channel dimension
        label = torch.from_numpy(label)

        return images, label