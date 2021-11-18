import torch.nn as nn
from gns import graph_network


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
      connectivity_radius,
      boundaries,
      normalization_stats,
      nparticle_types: int,
      particle_type_embedding_size,
      device="cpu",
  ):
    """Initializes the model.

      Args:
        particle_dimensions: Dimensionality of the problem.
        nnode_in: Number of node inputs.
        nedge_in: Number of edge inputs.
        latent_dim: Size of latent dimension (128)
        nmessage_passing_steps: Number of message passing steps.
        nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
        connectivity_radius: Scalar with the radius of connectivity.
        boundaries: List of 2-tuples, containing the lower and upper boundaries 
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
        nparticle_types, particle_type_embedding_size
    )

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=particle_dimensions,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    self._device = device
