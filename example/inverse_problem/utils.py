import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def compute_penalty(original_loss, threshold, alpha):
    """
    Computes a penalty term to ensure that loss does not go below the threshold.

    Args:
    - original_loss: Tensor, the computed loss.
    - threshold: float, the lower bound for the loss.
    - alpha: float, a hyperparameter to control the magnitude of the penalty.

    Returns:
    - penalty: Tensor, the computed penalty term.
    """
    penalty = torch.relu(threshold - original_loss) * alpha
    return penalty


def make_animation(
        positions, boundaries, output, particle_size=1.0, color="blue", timestep_stride=5):

    fig, ax = plt.subplots()

    def animate(i):
        fig.clear()
        # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=xboundary, ylim=yboundary)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
        ax.set_aspect("equal")
        ax.set_xlim([boundaries[0][0], boundaries[0][1]])
        ax.set_ylim([boundaries[1][0], boundaries[1][1]])
        ax.scatter(positions[i][:, 0], positions[i][:, 1], s=particle_size, c=color)
        ax.grid(True, which='both')

    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, len(positions), timestep_stride), interval=10)

    ani.save(output, dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {output}")


class To_Torch_Model_Param(torch.nn.Module):
    def __init__(self, parameters):
        super(To_Torch_Model_Param, self).__init__()
        self.current_params = torch.nn.Parameter(parameters)


def visualize_final_deposits(
        predicted_positions: torch.tensor,
        target_positions: torch.tensor,
        metadata: dict,
        write_path: str,
        friction=None,
):

    fig, ax = plt.subplots()
    inversion_positions_plot = predicted_positions.clone().detach().cpu().numpy()
    target_positions_plot = target_positions.clone().detach().cpu().numpy()
    ax.scatter(inversion_positions_plot[-1][:, 0],
               inversion_positions_plot[-1][:, 1], alpha=0.5, s=2.0, c="purple", label="Current")
    ax.axvline(x=inversion_positions_plot[-1][:, 0].max(), c="purple")
    ax.scatter(target_positions_plot[-1][:, 0],
               target_positions_plot[-1][:, 1], alpha=0.5, s=2.0, c="yellow", label="True")
    ax.axvline(x=target_positions_plot[-1][:, 0].max(), c="yellow")
    ax.set_xlim(metadata['bounds'][0])
    ax.set_ylim(metadata['bounds'][1])
    ax.set_title(f"{friction}:.5f")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True)
    plt.savefig(write_path)


def visualize_velocity_profile(
        predicted_velocities: torch.tensor,
        target_velocities: np.array,
        write_path
):
    fig, axes = plt.subplots(1, 2)
    n_vels = len(predicted_velocities)
    predicted_velocities = predicted_velocities.clone().detach().cpu().numpy()
    target_velocities = target_velocities

    # plot x vels
    axes[0].plot(range(n_vels), predicted_velocities[:, 0])
    axes[0].plot(range(n_vels), target_velocities[:, 0])
    axes[0].set_xlabel("Particle group id")
    axes[0].set_ylabel("x-velocity (m/s)")
    # plot y vels
    axes[1].plot(range(n_vels), predicted_velocities[:, 1])
    axes[1].plot(range(n_vels), target_velocities[:, 1])
    axes[0].set_xlabel("Particle group id")
    axes[0].set_ylabel("y-velocity (m/s)")
    plt.tight_layout()
    plt.savefig(write_path)

