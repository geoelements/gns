import pytest
import torch
import numpy as np
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from gns.learned_simulator import LearnedSimulator


def test_learned_simulator():
    # Define model parameters
    particle_dimensions = 2
    nnode_in = 30
    nedge_in = 3
    latent_dim = 128
    nmessage_passing_steps = 5
    nmlp_layers = 2
    mlp_hidden_dim = 64
    connectivity_radius = 1.0
    boundaries = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
    normalization_stats = {
        "acceleration": {"mean": torch.tensor([0.0, 0.0]), "std": torch.tensor([1.0, 1.0])},
        "velocity": {"mean": torch.tensor([0.0, 0.0]), "std": torch.tensor([1.0, 1.0])}
    }
    nparticle_types = 1
    particle_type_embedding_size = 16
    device = "cpu"

    # Instantiate model
    model = LearnedSimulator(
        particle_dimensions,
        nnode_in,
        nedge_in,
        latent_dim,
        nmessage_passing_steps,
        nmlp_layers,
        mlp_hidden_dim,
        connectivity_radius,
        boundaries.numpy(),
        normalization_stats,
        nparticle_types,
        particle_type_embedding_size,
        device
    )

    # Create a sequence of positions
    position_sequence = torch.tensor([[[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4], [1.5, 2.5]]])
    nparticles_per_example = torch.tensor([1])
    particle_types = torch.tensor([0])

    # Check if predict_positions works
    next_positions = model.predict_positions(
        position_sequence, nparticles_per_example, particle_types)
    assert next_positions.shape == (1, 2)

    # Check if predict_accelerations works
    predicted_normalized_acceleration, target_normalized_acceleration = model.predict_accelerations(
        next_positions, torch.tensor([0.1, 0.1]), position_sequence, nparticles_per_example, particle_types)
    assert predicted_normalized_acceleration.shape == (1, 2)
    assert target_normalized_acceleration.shape == (1, 2)

    # Check if the _inverse_decoder_postprocessor works
    next_position = torch.tensor([[1.6, 2.6]])
    normalized_acceleration = model._inverse_decoder_postprocessor(next_position, position_sequence)
    assert normalized_acceleration.shape == (1, 2)