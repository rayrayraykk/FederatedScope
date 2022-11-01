import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU


class ConvNet2(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True,
                 dropout=.0):
        """
        ConvNet2 class initialization.

        Args:
            self: write your description
            in_channels: write your description
            h: write your description
            w: write your description
            hidden: write your description
            class_num: write your description
            use_bn: write your description
            dropout: write your description
        """
        super(ConvNet2, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            self: write your description
            x: write your description
        """
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class ConvNet5(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 dropout=.0):
        """
        ConvNet5 class constructor.

        Args:
            self: write your description
            in_channels: write your description
            h: write your description
            w: write your description
            hidden: write your description
            class_num: write your description
            dropout: write your description
        """
        super(ConvNet5, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.bn1 = BatchNorm2d(32)

        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.bn2 = BatchNorm2d(64)

        self.conv3 = Conv2d(64, 64, 5, padding=2)
        self.bn3 = BatchNorm2d(64)

        self.conv4 = Conv2d(64, 128, 5, padding=2)
        self.bn4 = BatchNorm2d(128)

        self.conv5 = Conv2d(128, 128, 5, padding=2)
        self.bn5 = BatchNorm2d(128)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 128,
            hidden)
        self.fc2 = Linear(hidden, class_num)

        self.dropout = dropout

    def forward(self, x):
        """
        Forward pass of the RMRM module.

        Args:
            self: write your description
            x: write your description
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)

        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class VGG11(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=128,
                 class_num=10,
                 dropout=.0):
        """
        Initialize VGG11 class.

        Args:
            self: write your description
            in_channels: write your description
            h: write your description
            w: write your description
            hidden: write your description
            class_num: write your description
            dropout: write your description
        """
        super(VGG11, self).__init__()

        self.conv1 = Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = BatchNorm2d(64)

        self.conv2 = Conv2d(64, 128, 3, padding=1)
        self.bn2 = BatchNorm2d(128)

        self.conv3 = Conv2d(128, 256, 3, padding=1)
        self.bn3 = BatchNorm2d(256)

        self.conv4 = Conv2d(256, 256, 3, padding=1)
        self.bn4 = BatchNorm2d(256)

        self.conv5 = Conv2d(256, 512, 3, padding=1)
        self.bn5 = BatchNorm2d(512)

        self.conv6 = Conv2d(512, 512, 3, padding=1)
        self.bn6 = BatchNorm2d(512)

        self.conv7 = Conv2d(512, 512, 3, padding=1)
        self.bn7 = BatchNorm2d(512)

        self.conv8 = Conv2d(512, 512, 3, padding=1)
        self.bn8 = BatchNorm2d(512)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 512,
            hidden)
        self.fc2 = Linear(hidden, hidden)
        self.fc3 = Linear(hidden, class_num)

        self.dropout = dropout

    def forward(self, x):
        """
        Forward pass of the convolutional layer

        Args:
            self: write your description
            x: write your description
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn7(self.conv7(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn8(self.conv8(x)))
        x = self.maxpool(x)

        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x
