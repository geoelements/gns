from typing import List
import torch
import torch.nn as nn


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:
    """Build a MultiLayer Perceptron.

    Args:
      input_size: Size of input layer.
      layer_sizes: An array of input size for each hidden layer.
      output_size: Size of the output layer.
      output_activation: Activation function for the output layer.
      activation: Activation function for the hidden layers.

    Returns:
      mlp: An MLP sequential container.
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [activation for i in range(nlayers)]
    act[-1] = output_activation

    # Create a torch sequential container
    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                                 layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())
    return mlp


class Encoder(nn.Module):
    """Graph network encoder. Encode nodes and edges states to an MLP. The Encode: $\mathcal{X} \rightarrow \mathcal{G}$ embeds
    the particle-based state representation, $\mathcal{X}, as a latent graph, $G^0 = encoder(\mathcal{X})$, where 
    $G = (V, E, u)$, $v_i \in V$, and $e_{i,j} in E$"""

    def __init__(
            self,
            nnode_in_features: int,
            nnode_out_features: int,
            nedge_in_features: int,
            nedge_out_features: int,
            nmlp_layers: int,
            mlp_hidden_layer_size: int):
        """The Encoder implements nodes features $\varepsilon_v$ and edge features $\varepsilon_e$ as multilayer perceptrons (MLP) into the latent vectors, $v_i$ and $e_{i,j}$, of size 128.

        Args:
          nnode_in_features: Number of node input features (for 2D = 30, calculated as [10 = 5 times steps * 2 positions (x, y) + 4 distances to boundaries (top/bottom/left/right) + 16 particle type embeddings]).
          nnode_out_features: Number of node output features (particle dimension).
          nedge_in_features: Number of edge input features (for 2D = 3, calculated as [2 (x, y) relative displacements between 2 particles + distance between 2 particles]).
          nedge_out_features: Number of edge output features (latent dimension size of 128).
          nmlp_layer: Number of hidden layers in the MLP.
          mlp_hidden_dim: Size of the hidden layer.

        Returns:
          mlp: An MLP sequential container.
        """
        super(Encoder, self).__init__()
        # Encode node features as an MLP
        self.node_fn = nn.Sequential(*[build_mlp(nnode_in_features, [mlp_hidden_layer_size for _ in range(nmlp_layers)], nnode_out_features),
                                       nn.LayerNorm(nnode_out_features)])
        # Encode edge features as an MLP
        self.edge_fn = nn.Sequential(*[build_mlp(nedge_in_features, [mlp_hidden_layer_size for _ in range(nmlp_layers)], nedge_out_features),
                                       nn.LayerNorm(nedge_out_features)])

    def forward(
            self,
            x: torch.tensor,
            edge_features: torch.tensor):
        """Forward hook runs when the class is instantiated

        Args:
          x: Particle state representation as a torch tensor with shape (nparticles, nnode_input_features) 
          edge_features: Edge features as a torch tensor with shape (nparticles, nedge_input_features) 
        """
        return self.node_fn(x), self.edge_fn(edge_features)
