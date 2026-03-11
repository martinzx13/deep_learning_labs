import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearRegressionModel, self).__init__()
        # Use the arguments to make the layer flexible
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The forward pass is now just one clean call
        return self.linear(x)