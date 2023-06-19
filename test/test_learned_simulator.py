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
    particle_dimensions = 2
    nnode_in = 4
    nedge_in = 5
    latent_dim = 16
    nmessage_passing_steps = 5
    nmlp_layers = 2
    mlp_hidden_dim = 64
    connectivity_radius = 0.2
    boundaries = np.array([[0., 1.], [0., 1.]])
    normalization_stats = {"acceleration": {'mean': 0., 'std': 1.},
                           "velocity": {'mean': 0., 'std': 1.}}
    nparticle_types = 2
    particle_type_embedding_size = 16
    device = "cpu"
    return LearnedSimulator(
        particle_dimensions, nnode_in, nedge_in, latent_dim,
        nmessage_passing_steps, nmlp_layers, mlp_hidden_dim,
        connectivity_radius, boundaries, normalization_stats, nparticle_types,
        particle_type_embedding_size, device)


def test_predict_positions(simulator):
    current_positions = torch.tensor([[[0., 0.], [0., 0.1], [0., 0.2], [0., 0.3], [0., 0.4], [0., 0.5]],
                                      [[0., 1.], [0., 1.1], [0., 1.2], [0., 1.3], [0., 1.4], [0., 1.5]]])
    nparticles_per_example = torch.tensor([2])
    particle_types = torch.tensor([0, 1])
    next_positions = simulator.predict_positions(current_positions, nparticles_per_example, particle_types)
    assert next_positions.shape == current_positions.shape
    assert next_positions.dtype == current_positions.dtype


def test_save_load(simulator):
    current_positions = torch.tensor([[[0., 0.], [0., 0.1], [0., 0.2], [0., 0.3], [0., 0.4], [0., 0.5]],
                                      [[0., 1.], [0., 1.1], [0., 1.2], [0., 1.3], [0., 1.4], [0., 1.5]]])
    nparticles_per_example = torch.tensor([2])
    particle_types = torch.tensor([0, 1])
    path = "test_model.pt"

    # Save the model
    simulator.save(path)
    # Load the model
    simulator.load(path)
    # Run predictions
    next_positions = simulator.predict_positions(current_positions, nparticles_per_example, particle_types)
    assert next_positions.shape == current_positions.shape
    assert next_positions.dtype == current_positions.dtype

