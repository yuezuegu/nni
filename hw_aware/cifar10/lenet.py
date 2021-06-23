'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch

def calculate_size(layers, input_dim):
    x = torch.randn(input_dim)
    for l in layers:
        x = l(x)
    return torch.flatten(x,start_dim=1).shape[1]

class LeNet(nn.Module):
    def __init__(self, params, input_dim):
        super(LeNet, self).__init__()

        conv1_out_channels = int(params["conv1_out_channels"])
        conv2_out_channels = int(params["conv2_out_channels"])
        fc1_out_channels = int(params["fc1_out_channels"])
        fc2_out_channels = int(params["fc2_out_channels"])
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_out_channels, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(        
            nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(calculate_size([self.conv1, self.conv2], input_dim), fc1_out_channels),
            nn.ReLU(inplace=True)
        )

        self.fc2   = nn.Sequential(
            nn.Linear(fc1_out_channels, fc2_out_channels),
            nn.ReLU(inplace=True)
        )

        self.fc3   = nn.Sequential(
            nn.Linear(fc2_out_channels, 10),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x