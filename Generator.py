import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size[0])
        self.map2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.map3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.map4 = nn.Linear(hidden_size[2], output_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x), 0.2)
        x = F.leaky_relu(self.map2(x), 0.2)
        x = F.leaky_relu(self.map3(x), 0.2)
        x = F.tanh(self.map4(x))
        x = x.view(-1, 1, 16, 16)
        return x