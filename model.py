import numpy as np
import torch
from torch import nn

class DropoutLayer(nn.Module):
    def __init__(self, hidden_dim, n_particle=1):
        super(DropoutLayer, self).__init__()
        self.mask = torch.ones(size=(n_particle,hidden_dim), dtype=torch.float32, requires_grad=False)
        self.training = False

    def forward(self, x):
        return x * self.mask

    def update(self, mask: np.ndarray):
        assert mask.shape == self.mask.shape, f"new mask shape should be {self.mask.shape} but giving {mask.shape}"
        self.mask = torch.Tensor(mask)
        
    def get(self):
        return self.mask.numpy()

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                output_dim, n_particle=1):
        super().__init__()
        self.lin1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.lin3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.relu = nn.ReLU()
        # dropout for pretraining and finetunning
        self.dropout_for_GD = nn.Dropout(p=0.5)
        # dropout for smc
        self.dropout = DropoutLayer(hidden_dim=hidden_dim , n_particle=n_particle)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        if self.training:
            x = self.dropout_for_GD(x)
        else:
            x = self.dropout(x)
        x = self.lin3(x)
        return x
        
    def update_dropout_mask(self, mask: np.ndarray):
        self.dropout.update(mask)

    def get_dropout_mask(self):
        return self.dropout.get()