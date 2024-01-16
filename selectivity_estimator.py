from collections import namedtuple

import numpy
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import misc_utils

class Cardinality_Estimator(nn.Module):
    """
    Parameters:
        num_layers (int): Number of GRU layers in the model.
        hidden_size (int): Number of features in the hidden state of the GRU.
        device (torch.device): Device (CPU/GPU) on which the model will be loaded and computations will be performed.
        alphabet_size (int): Size of the input alphabet (number of different symbols).

    Attributes:
        num_layers (int): Number of GRU layers in the model.
        hidden_size (int): Number of features in the hidden state of the GRU.
        device (torch.device): Device (CPU/GPU) on which the model will be loaded and computations will be performed.
        alphabet_size (int): Size of the input alphabet (number of different symbols).
        layer_sizes (list): List of integers representing the sizes of fully connected layers in the model.
        gru (nn.GRU): GRU layer of the model.
        fc (nn.Sequential): Fully connected neural network with ReLU activation.
    """
    def __init__(self, num_layers, hidden_size, device, alphabet_size):
        super(Cardinality_Estimator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device  = device
        self.alphabet_size =alphabet_size
        layer_sizes = [128, 64, 32, 16, 8]
        self.gru = nn.GRU(
            batch_first=True,
            input_size=self.alphabet_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.Linear(layer_sizes[4], 1),
        )


    def forward_selectivity(self, x):
        output, hidden_state = self.gru(x)
        output = torch.sigmoid(self.fc(output))
        return torch.squeeze(output)

    def forward(self, x):
        output = self.forward_selectivity(x)
        return output

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)

def train_model(train_data, model, device, learning_rate, num_epocs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    model.train()
    for epoch in range(num_epocs):
        loss_list = []
        for i, (name, mask, target) in enumerate(train_data):
            name = name.to(device)
            output = model(name)
            output = output.to(device)
            target = target.to(device)
            mask = mask.to(device)
            loss = misc_utils.binary_crossentropy(output, target, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss)
        print("Epoch: {}/{} - Mean Running Loss: {:.4f}".format(epoch+1, num_epocs, np.mean(numpy.array( torch.tensor(loss_list, device = 'cpu')))))

    return model
