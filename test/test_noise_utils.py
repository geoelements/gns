import pytest
import torch
import numpy as np
import os, sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from gns.noise_utils import get_random_walk_noise_for_position_sequence
from gns.learned_simulator import time_diff

@pytest.mark.parametrize("nparticles, dim, noise_std_last_step", [(10, 2, 0.1), (20, 3, 0.5)])
def test_get_random_walk_noise_for_position_sequence(nparticles, dim, noise_std_last_step):
    position_sequence = torch.randn(nparticles, 6, dim)
    position_sequence_noise = get_random_walk_noise_for_position_sequence(
        position_sequence, noise_std_last_step)

    assert position_sequence_noise.shape == position_sequence.shape, \
        "Output tensor has incorrect shape."

    velocity_sequence_noise = time_diff(position_sequence_noise)
    # Compute the standard deviation of the noise in the last velocity
    computed_noise_std_last_step = torch.std(velocity_sequence_noise[:, -1, :])
    print("computed_noise_std_last_step: ", computed_noise_std_last_step)
    print("noise_std_last_step: ", noise_std_last_step)

    """
    np.testing.assert_allclose(
        computed_noise_std_last_step.numpy(), 
        noise_std_last_step, 
        atol=1e-3, 
        err_msg="Standard deviation of the noise in the last step does not match the input."
    )
    """

    # Check that the first position has no noise
    assert torch.all(position_sequence_noise[:, 0, :] == 0), \
        "The first position is expected to have no noise."
