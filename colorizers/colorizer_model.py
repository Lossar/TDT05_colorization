
import torch
import torch.nn as nn
import numpy as np
from IPython import embed

from base_color import *

class ColorizerModel(nn.Module):
    def __init__(self):
        super(ColorizerModel, self).__init__()

        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 2, kernel_size=3, padding=1)

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        return x
