import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import animation


flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout `.pkl` file")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.")
FLAGS = flags.FLAGS

def render_gif_animation():

    rollout_path = f"{FLAGS.rollout_dir}/{FLAGS.rollout_name}.pkl"
    animation_filename = f"{FLAGS.rollout_dir}/{FLAGS.rollout_name}.gif"

    # read rollout data
    with open(rollout_path, 'rb') as f:
        result = pickle.load(f)
    ground_truth_vel = np.concatenate((result["initial_velocities"], result["ground_truth_rollout"]))
    predicted_vel = np.concatenate((result["initial_velocities"], result["predicted_rollout"]))

    # compute velocity magnitude
    ground_truth_vel_magnitude = np.linalg.norm(ground_truth_vel, axis=-1)
    predicted_vel_magnitude = np.linalg.norm(predicted_vel, axis=-1)
    velocity_result = {
        "ground_truth": ground_truth_vel_magnitude,
        "prediction": predicted_vel_magnitude
    }

    # variables for render
    n_timesteps = len(ground_truth_vel_magnitude)
    triang = tri.Triangulation(result["node_coords"][0][:, 0], result["node_coords"][0][:, 1])

    # color
    vmin = np.concatenate(
        (result["predicted_rollout"][0][:, 0], result["ground_truth_rollout"][0][:, 0])).min()
    vmax = np.concatenate(
        (result["predicted_rollout"][0][:, 0], result["ground_truth_rollout"][0][:, 0])).max()

    # Init figures
    fig = plt.figure(figsize=(9.75, 3))

    def animate(i):
        print(f"Render step {i}/{n_timesteps}")

        fig.clear()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 1),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="1.5%",
                         cbar_pad=0.15)

        for j, (sim, vel) in enumerate(velocity_result.items()):
            grid[j].triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)
            handle = grid[j].tripcolor(triang, vel[i], vmax=vmax, vmin=vmin)
            fig.colorbar(handle, cax=grid.cbar_axes[0])
            grid[j].set_title(sim)

    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, n_timesteps, FLAGS.step_stride), interval=20)

    ani.save(f'{animation_filename}', dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {animation_filename}")


def main(_):
    render_gif_animation()


if __name__ == '__main__':
    app.run(main)
