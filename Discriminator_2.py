import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size[0])
        self.map2 = nn.Linear(hidden_size[0], output_size)
        #self.map3 = nn.Linear(hidden_size[1], output_size)

    def forward(self, x):
        x = x.view(-1, 16 * 16)
        x = F.leaky_relu(self.map1(x), 0.2)
        #x = F.leaky_relu(self.map2(x), 0.2)
        x = F.sigmoid(self.map2(x))
        return x