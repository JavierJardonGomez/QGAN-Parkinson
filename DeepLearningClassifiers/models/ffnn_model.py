import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class FFNNClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(FFNNClassifier, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))  # 8000 -> 128
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))  # 128 -> 128
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))  # 128 -> 2 (salida)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



