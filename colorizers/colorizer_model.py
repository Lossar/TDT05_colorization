
import torch
import torch.nn as nn
import numpy as np
from IPython import embed

from base_color import *

class ColorizerModel(BaseColor):
    def __init__(self):
        super(ColorizerModel, self).__init__()

        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.model_out = nn.Conv2d(64, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.model_out(self.softmax(x))
        return x
