import torch
import torch.nn as nn

class XORModel(nn.Module):
  def __init__(self, input_dim:int, output_dim:int, hidden_dim:int):
    super(XORModel, self).__init__()
    self.hidden = nn.Linear(input_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.output = nn.Linear(hidden_dim, output_dim)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x:torch.Tensor)->torch.Tensor:
    x = self.hidden(x)
    x = self.relu(x)
    x = self.output(x)
    x = self.sigmoid(x)
    return (x)