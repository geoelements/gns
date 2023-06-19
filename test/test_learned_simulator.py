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
def model():
    particle_dimensions = 2
    nnode_in = 30
    nedge_in = 3
    latent_dim = 128
    nmessage_passing_steps = 2
    nmlp_layers = 2
    mlp_hidden_dim = 128
    connectivity_radius = 0.1
    boundaries = np.array([[0.0, 1.0], [0.0, 1.0]])
    normalization_stats = {
        "acceleration": {"mean": 0.0, "std": 1.0},
        "velocity": {"mean": 0.0, "std": 1.0}
    }
    nparticle_types = 1
    particle_type_embedding_size = 16
    device = "cpu"

    return LearnedSimulator(
        particle_dimensions=particle_dimensions,
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        connectivity_radius=connectivity_radius,
        boundaries=boundaries,
        normalization_stats=normalization_stats,
        nparticle_types=nparticle_types,
        particle_type_embedding_size=particle_type_embedding_size,
        device=device
    )


@pytest.fixture
def particles():
    current_positions = torch.tensor([[0.5, 0.5], [0.6, 0.6]])
    position_sequence = torch.tensor(
        [
            [[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
            [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]
        ]
    )
    nparticles_per_example = torch.tensor([2])
    particle_types = torch.tensor([0, 0])

    return current_positions, position_sequence, nparticles_per_example, particle_types


def test_predict_positions(model, particles):
    current_positions, position_sequence, nparticles_per_example, particle_types = particles

    next_positions = model.predict_positions(
        current_positions, nparticles_per_example, particle_types)

    assert next_positions.shape == current_positions.shape
    assert next_positions.dtype == current_positions.dtype

    # Some basic checks to ensure the positions are as expected. In reality,
    # more thorough checks might be needed depending on the expected behavior
    # of the model.
    assert torch.all(next_positions >= 0.0)
    assert torch.all(next_positions <= 1.0)
