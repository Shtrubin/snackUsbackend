import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)  
        self.b1 = nn.Parameter(torch.zeros(hidden_size)) 
        self.w2 = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01) 
        self.b2 = nn.Parameter(torch.zeros(hidden_size)) 
        self.w3 = nn.Parameter(torch.randn(hidden_size, num_classes) * 0.01) 
        self.b3 = nn.Parameter(torch.zeros(num_classes)) 
    
    def relu(self, x):
        return torch.maximum(x, torch.zeros_like(x)) 
    
    def forward(self, x):
        out1 = torch.matmul(x, self.w1) + self.b1
        out1 = self.relu(out1)
        
        out2 = torch.matmul(out1, self.w2) + self.b2
        out2 = self.relu(out2)
        
        out3 = torch.matmul(out2, self.w3) + self.b3
        return out3
