import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out



# Why Not Other Networks?
# RNN (Recurrent Neural Network) would require recurrence (loops where the output of a layer is fed back into the network).
# CNN (Convolutional Neural Network) would involve convolutional layers that are specialized for image processing tasks.
# Since this model uses only linear layers and ReLU activations, it's a feedforward neural network.