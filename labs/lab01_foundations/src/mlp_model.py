import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
    super(MLPRegressor, self).__init__()
    self.hidden = nn.Linear(input_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.output = nn.Linear(hidden_dim, output_dim)
  def forward(self, x:torch.Tensor)->torch.Tensor:
    x = self.hidden(x)
    x = self.relu(x)
    x = self.output(x)
    return (x)
  
class DeepMLRegressor(nn.Module):
  def __init__(self, input_dim:int, h1_dim:int, h2_dim:int, output_dim:int):
    super(DeepMLRegressor, self).__init__()
    self.layer_1 = nn.Linear(input_dim, h1_dim)
    self.layer_2 = nn.Linear(h1_dim, h2_dim)
    self.layer_output = nn.Linear(h2_dim, output_dim)
    self.relu = nn.ReLU()

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    x = self.relu(self.layer_1(x))
    x = self.relu(self.layer_2(x))
    x = self.layer_output(x)
    return (x)
