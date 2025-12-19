import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_channels, 3, kernel_size=1)
    def forward(self, a):
        a = F.relu(self.conv1(a))
        a = F.relu(self.conv2(a))
        a = F.relu(self.conv3(a))
        out = self.conv_out(a)
        return out

