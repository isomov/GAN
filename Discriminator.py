import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers=nn.Sequential(nn.Conv2d(1, 16, 4, padding=2),
                                nn.MaxPool2d(2, 2),
                                nn.ReLU(),
                                nn.Conv2d(16, 32, 3, padding=1),
                                nn.MaxPool2d(2, 2),
                                nn.ReLU())
        self.fc1 = nn.Linear(4 * 4 * 32, 1)
    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 4 * 4 * 32)
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x