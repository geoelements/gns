import pytest
import torch
import numpy as np
import os, sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from gns.noise_utils import get_random_walk_noise_for_position_sequence
from gns.learned_simulator import time_diff

def test_get_random_walk_noise_for_position_sequence():
    # Define input tensor for position_sequence
    position_sequence = torch.rand((10, 6, 3))

    # Define standard deviation for last step
    noise_std_last_step = 0.2

    # Call the function with the defined inputs
    output = get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step)

    # Assert that the function doesn't fail and returns a tensor of the same shape
    assert output.shape == position_sequence.shape

    # Assert that the standard deviation of the last step is close to the desired value
    # We use np.isclose to deal with floating point imprecision
    assert np.isclose(output[:, -1].std().item(), noise_std_last_step, atol=0.05)

    # Check that the first position has no noise
    assert torch.all(output[:, 0, :] == 0), \
        "The first position is expected to have no noise."

@pytest.mark.parametrize("shape", [
    (10, 6, 3),
    (1, 5, 3),
    (100, 7, 2)
])
def test_shapes(shape):
    position_sequence = torch.rand(shape)
    noise_std_last_step = 0.1
    output = get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step)
    assert output.shape == position_sequence.shape
