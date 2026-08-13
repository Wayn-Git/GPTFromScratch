import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        with torch.no_grad():

        # linear_layers = 
           stats = list()

           for layer in model:
               x = layer(x)
               if isinstance(layer, nn.Linear):
                
                  mean_val = torch.mean(x).item()
                
                  std_val = torch.std(x).item()

                  total_neurons = x.numel()
                  dead_neruons = torch.sum(x == 0.0).item()
                  dead_fraction = dead_neruons / total_neurons
                  

                  stats.append({'mean': round(mean_val,4), 'std': round(std_val, 4), 'dead_fraction': round(dead_fraction, 4)})

        return stats




    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        pass

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        pass
