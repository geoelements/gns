import torch
from gns import learned_simulator


def get_random_walk_noise_for_position_sequence(
    position_sequence: torch.tensor, noise_std_last_step
):
    """Returns random-walk noise in the velocity applied to the position.

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      noise_std_last_step: Standard deviation of noise in the last step.

    """
    velocity_sequence = learned_simulator.time_diff(position_sequence)

    # We want the noise scale in the velocity at the last step to be fixed.
    # Because we are going to compose noise at each step using a random_walk:
    # std_last_step**2 = num_velocities * std_each_step**2
    # so to keep `std_last_step` fixed, we apply at each step:
    # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
    num_velocities = velocity_sequence.shape[1]

    # Optimized: Create noise directly on same device as input
    velocity_sequence_noise = torch.randn(
        velocity_sequence.shape,
        device=position_sequence.device,
        dtype=position_sequence.dtype
    ) * (noise_std_last_step / num_velocities**0.5)

    # Apply the random walk.
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    # Integrate the noise in the velocity to the positions, assuming
    # an Euler integrator and a dt = 1, and adding no noise to the very first
    # position (since that will only be used to calculate the first position
    # change).
    # Optimized: Pre-allocate on correct device
    position_sequence_noise = torch.zeros(
        (velocity_sequence.shape[0], velocity_sequence.shape[1] + 1, velocity_sequence.shape[2]),
        device=position_sequence.device,
        dtype=position_sequence.dtype
    )
    position_sequence_noise[:, 1:] = torch.cumsum(velocity_sequence_noise, dim=1)

    return position_sequence_noise
