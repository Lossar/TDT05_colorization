
import torch
import torch.nn as nn
import numpy as np
from IPython import embed

from base_color import *

class ColorizerModel(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ColorizerModel, self).__init__()

        # Define the layers of the CNN
        self.conv1_0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2_0 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv3_0 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.conv4_0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm2d(256)

        self.conv5_0 = nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)
        self.relu = nn.ReLU()
        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)
        self.relu = nn.ReLU()
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)
        self.relu = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm2d(256)

        self.conv6_0 = nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)
        self.relu = nn.ReLU()
        self.conv6_1 = nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)
        self.relu = nn.ReLU()
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)
        self.relu = nn.ReLU()
        self.batchnorm6 = nn.BatchNorm2d(256)

        self.conv7_0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv7_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.batchnorm7 = nn.BatchNorm2d(256)

        self.conv8_0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv8_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        
        self.conv8_3 = nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1_0(x)
        x = self.relu(x)
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)

        x = self.conv2_0(x)
        x = self.relu(x)
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.batchnorm2(x)

        x = self.conv3_0(x)
        x = self.relu(x)
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.batchnorm3(x)

        x = self.conv4_0(x)
        x = self.relu(x)
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.batchnorm4(x)

        x = self.conv5_0(x)
        x = self.relu(x)
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.batchnorm5(x)

        x = self.conv6_0(x)
        x = self.relu(x)
        x = self.conv6_1(x)
        x = self.relu(x)
        x = self.conv6_2(x)
        x = self.relu(x)
        x = self.batchnorm6(x)

        x = self.conv7_0(x)
        x = self.relu(x)
        x = self.conv7_1(x)
        x = self.relu(x)
        x = self.conv7_2(x)
        x = self.relu(x)
        x = self.batchnorm7(x)

        x = self.conv8_0(x)
        x = self.relu(x)
        x = self.conv8_1(x)
        x = self.relu(x)
        x = self.conv8_2(x)
        x = self.relu(x)
        x = self.conv8_3(x)

        out = self.model_out(self.softmax(x))
        return self.upsample4(out)
