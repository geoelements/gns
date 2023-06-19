import pytest
import torch
import numpy as np
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from gns.learned_simulator import LearnedSimulator

@pytest.fixture
def simulator():
    """Fixture to create a simulator object"""
    particle_dimensions = 2
    nnode_in = 30
    nedge_in = 3
    latent_dim = 128
    nmessage_passing_steps = 5
    nmlp_layers = 2
    mlp_hidden_dim = 64
    connectivity_radius = 0.05
    boundaries = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    normalization_stats = {
        "acceleration": {'mean': 0., 'std': 1.},
        "velocity": {'mean': 0., 'std': 1.}
    }
    nparticle_types = 1
    particle_type_embedding_size = 16
    device = "cpu"

    return LearnedSimulator(
        particle_dimensions, nnode_in, nedge_in, latent_dim, nmessage_passing_steps,
        nmlp_layers, mlp_hidden_dim, connectivity_radius, boundaries,
        normalization_stats, nparticle_types, particle_type_embedding_size, device)


def test_encoder_preprocessor(simulator):
    """Test for _encoder_preprocessor"""
    position_sequence = torch.tensor([[[0.0, 0.0], [0.0, 0.1], [0.0, 0.2], [0.0, 0.3], [0.0, 0.4], [0.0, 0.5]],
                                      [[0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0]]])
    nparticles_per_example = torch.tensor([2])
    particle_types = torch.tensor([0, 0])

    node_features, edge_index, edge_features = simulator._encoder_preprocessor(
        position_sequence, nparticles_per_example, particle_types)

    assert node_features.shape == (2, 14)
    assert edge_index.shape == (2, 2)  # one edge between the 2 particles
    assert edge_features.shape == (2, 3)  # one edge with 3 features

    # check the constructed graph has the expected nodes
    assert set(edge_index[0].tolist()) == set([0, 1])  # senders
    assert set(edge_index[1].tolist()) == set([0, 1])  # receivers

    # check the constructed graph has the expected edge
    assert {tuple(edge_index[:, i].tolist()) for i in range(edge_index.shape[1])} == {(0, 1), (1, 0)}
