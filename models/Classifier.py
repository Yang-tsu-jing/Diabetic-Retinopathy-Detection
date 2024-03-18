import torch
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):
    def __init__(self, in_channels, num_class):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 128)
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(128, num_class)
    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
        