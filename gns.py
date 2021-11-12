from typing import List

import torch
import torch.nn as nn


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: torch.nn.Module = torch.nn.Identity,
        activation: torch.nn.Module = torch.nn.ReLU) -> torch.nn.Module:
    """Build a MultiLayer Perceptron.

    Args:
      input_size: Size of input layer.
      layer_sizes: An array of input size for each hidden layer.
      output_size: Size of the output layer.
      output_activation: Activation function for the output layer.
      activation: Activation function for the hidden layers.

    Returns:
      torch.nn.Sequential: A sequantial container.
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    acts = [activation for i in range(nlayers)]
    acts[-1] = output_activation

    # Create a torch sequential container
    layers = [[torch.nn.Linear(layer_sizes[i],
                               layer_sizes[i + 1]), acts[i]()]
              for i in range(nlayers)]
    return torch.nn.Sequential(*layers)
