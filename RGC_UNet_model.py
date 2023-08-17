import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



# Define the general dense block
class DenseBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(DenseBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 24, 3, padding=1, bias=False),
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(in_channels + 24),
            nn.ReLU(),
            nn.Conv2d(in_channels + 24, 24, 3, padding=1, bias=False),
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(in_channels + 2 * 24),
            nn.ReLU(),
            nn.Conv2d(in_channels + 2 * 24, 24, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(torch.cat([x, x1], 1))
        x3 = self.layer3(torch.cat([x, x1, x2], 1))
        return torch.cat([x, x1, x2, x3], 1)


# Define the downsample block
class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, drop_rate: float = 0.0):
        super(DownSample, self).__init__()
        self.conv_block_pool = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(),
            DenseBlock(out_channels),
            nn.Dropout2d(p=drop_rate),
        )

        nn.init.kaiming_normal_(self.conv_block_pool[0].weight, nonlinearity='relu')

    def forward(self, x):
        return self.conv_block_pool(x)
    

# Define the upsample block
class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, drop_rate: float = 0.0):
        super(UpSample, self).__init__()
        self.conv_block_up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(),
            DenseBlock(out_channels),
            nn.Dropout2d(p=drop_rate),
            nn.Upsample(scale_factor=2),
        )

        nn.init.kaiming_normal_(self.conv_block_up[0].weight, nonlinearity='relu')

    def forward(self, x):
        return self.conv_block_up(x)
    

# Define the output block
class OutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutputBlock, self).__init__()
        self.conv_block_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(),
            DenseBlock(out_channels),
            nn.Conv2d(out_channels + 3 * 24, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 1, 1),
            nn.Sigmoid()
        )

        nn.init.kaiming_normal_(self.conv_block_out[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_block_out[3].weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.conv_block_out[5].weight)

    def forward(self, x):
        return self.conv_block_out(x)


# Define the RGC_UNet model
class RGC_UNet(nn.Module):
    def __init__(self):
        super(RGC_UNet, self).__init__()
        self.down1 = DownSample(in_channels=1, out_channels=64)
        self.down2 = DownSample(in_channels=136, out_channels=128)
        self.down3 = DownSample(in_channels=200, out_channels=256)
        self.down4 = DownSample(in_channels=328, out_channels=512, drop_rate=0.5)
        self.up5 = UpSample(in_channels=584, out_channels=1024, drop_rate=0.5)
        self.up6 = UpSample(in_channels=1096, out_channels=512)
        self.up7 = UpSample(in_channels=584, out_channels=256)
        self.up8 = UpSample(in_channels=328, out_channels=128)
        self.conv5 = nn.Conv2d(1096, 512, 2, padding='same')
        self.conv6 = nn.Conv2d(584, 256, 2, padding='same')
        self.conv7 = nn.Conv2d(328, 128, 2, padding='same')
        self.conv8 = nn.Conv2d(200, 64, 2, padding='same')
        self.out9 = OutputBlock(in_channels=200, out_channels=64)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        block1 = self.down1(x)
        maxpool1 = self.maxpool(block1)
        block2 = self.down2(maxpool1)
        maxpool2 = self.maxpool(block2)
        block3 = self.down3(maxpool2)
        maxpool3 = self.maxpool(block3)
        block4 = self.down4(maxpool3)
        maxpool4 = self.maxpool(block4)
        block5 = self.up5(maxpool4)
        upsample5 = self.relu(self.conv5(block5))
        block6 = self.up6(torch.cat([block4, upsample5], 1))
        upsample6 = self.relu(self.conv6(block6))
        block7 = self.up7(torch.cat([block3, upsample6], 1))
        upsample7 = self.relu(self.conv7(block7))
        block8 = self.up8(torch.cat([block2, upsample7], 1))
        upsample8 = self.relu(self.conv8(block8))
        image_prediction = self.out9(torch.cat([block1, upsample8], 1))

        return image_prediction

# # Test to see if an image can be passed through the model
# image = torch.rand(1, 1, 256, 256)
# model = RGC_UNet()

# # Print the size of the output
# print(model(image).size())

# Print the total number of parameters
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))