import torch
import torch.nn as nn
import numpy as np
from gns import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict


class LearnedSimulator(nn.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
          self,
          particle_dimensions: int,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          boundaries: np.ndarray,
          normalization_stats: Dict,
          nparticle_types: int,
          particle_type_embedding_size,
          device="cpu"):
    """Initializes the model.

    Args:
      particle_dimensions: Dimensionality of the problem.
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      connectivity_radius: Scalar with the radius of connectivity.
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      device: Runtime device (cuda or cpu).

    """
    super(LearnedSimulator, self).__init__()
    self._boundaries = boundaries
    self._connectivity_radius = connectivity_radius
    self._normalization_stats = normalization_stats
    self._nparticle_types = nparticle_types

    # Particle type embedding has shape (9, 16)
    self._particle_type_embedding = nn.Embedding(
        nparticle_types, particle_type_embedding_size)

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=particle_dimensions,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)

    self._device = device

  def forward(self):
    """Forward hook runs on class instantiation"""
    pass

  def _compute_graph_connectivity(
          self,
          node_features: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = True):
    """Generate graph edges to all points within a threshold radius

    Args:
      node_features: Node features with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      radius: Threshold to construct edges to all points within the radius.
      add_self_edges: Boolean flag to include self edge (default: True)
    """
    # Specify examples id for particles
    batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate(nparticles_per_example)]).to(self._device)

    # radius_graph accepts r < radius not r <= radius
    # A torch tensor list of source and target nodes with shape (2, nedges)
    edge_index = radius_graph(
        node_features, r=radius, batch=batch_ids, loop=add_self_edges)

    # The flow direction when using in combination with message passing is
    # "source_to_target"
    receivers = edge_index[0, :]
    senders = edge_index[1, :]

    return receivers, senders

  def _encoder_preprocessor(
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor):
    """Extracts important features from the position sequence. Returns a tuple 
    of node_features (nparticles, 30), edge_index (nparticles, nparticles), and 
    edge_features (nparticles, 3).

    Args:
      position_sequence: A sequence of particle positions. Shape is 
        (nparticles, 6, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles)
    """
    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1]  # (n_nodes, 2)
    velocity_sequence = time_diff(position_sequence)

    # Get connectivity of the graph with shape of (nparticles, 2)
    senders, receivers = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)
    node_features = []

    # Normalized velocity sequence, merging spatial an time axis.
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
    flat_velocity_sequence = normalized_velocity_sequence.view(
        nparticles, -1)
    # There are 5 previous steps, with dim 2
    # node_features shape (nparticles, 5 * 2 = 10)
    node_features.append(flat_velocity_sequence)

    # Normalized clipped distances to lower and upper boundaries.
    # boundaries are an array of shape [num_dimensions, 2], where the second
    # axis, provides the lower/upper boundaries.
    boundaries = torch.tensor(
        self._boundaries, requires_grad=False).float().to(self._device)
    distance_to_lower_boundary = (
        most_recent_position - boundaries[:, 0][None])
    distance_to_upper_boundary = (
        boundaries[:, 1][None] - most_recent_position)
    distance_to_boundaries = torch.cat(
        [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
    normalized_clipped_distance_to_boundaries = torch.clamp(
        distance_to_boundaries / self._connectivity_radius, -1., 1.)
    # The distance to 4 boundaries (top/bottom/left/right)
    # node_features shape (nparticles, 10+4)
    node_features.append(normalized_clipped_distance_to_boundaries)

    # Particle type
    if self._nparticle_types > 1:
      particle_type_embeddings = self._particle_type_embedding(
          particle_types)
      node_features.append(particle_type_embeddings)
    # Final node_features shape (nparticles, 30) for 2D
    # 30 = 10 (5 velocity sequences*dim) + 4 boundaries + 16 particle embedding

    # Collect edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius
    # with shape (nedges, 2)
    # normalized_relative_displacements = (
    #     torch.gather(most_recent_position, 0, senders) -
    #     torch.gather(most_recent_position, 0, receivers)
    # ) / self._connectivity_radius
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius

    # Add relative displacement between two particles as an edge feature
    # with shape (nparticles, ndim)
    edge_features.append(normalized_relative_displacements)

    # Add relative distance between 2 particles with shape (nparticles, 1)
    # Edge features has a final shape of (nparticles, ndim + 1)
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))


def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
  """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
  return position_sequence[:, 1:] - position_sequence[:, :-1]
