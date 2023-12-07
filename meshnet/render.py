import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import animation
from utils import NodeType


flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout `.pkl` file")
flags.DEFINE_string("mesh_type", "quad", help="Mesh type, either `triangle` or `quad`")
flags.DEFINE_integer("nnode_x", 161, help="Argument only for quad mesh render. Number of nodes for x-axis")
flags.DEFINE_integer("nnode_y", 41, help="Argument only for quad mesh render. Number of nodes for y-axis")
flags.DEFINE_integer("step_stride", 5, help="Stride of steps to skip.")
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
    if FLAGS.mesh_type == "triangle":
        triang = tri.Triangulation(result["node_coords"][0][:, 0], result["node_coords"][0][:, 1])


    # color
    # TODO (yc): need to determine a single color bar parameter.
    vmin = np.concatenate(
        (result["predicted_rollout"][0][:, 0], result["ground_truth_rollout"][0][:, 0])).min()
    vmax = np.concatenate(
        (result["predicted_rollout"][0][:, 0], result["ground_truth_rollout"][0][:, 0])).max()
    # Setting contour levels between min and max values of magnitudes
    levels = np.linspace(velocity_result["ground_truth"].min(),
                         velocity_result["ground_truth"].max(),
                         100)

    # Create masks for each node type
    node_type = result["node_types"][0]
    mask_normal = np.squeeze(node_type == NodeType.NORMAL)
    mask_outflow = np.squeeze(node_type == NodeType.OUTFLOW)
    mask_inflow = np.squeeze(node_type == NodeType.INFLOW)
    mask_wall = np.squeeze(node_type == NodeType.WALL_BOUNDARY)

    # Init figures
    fig = plt.figure(figsize=(7, 4))

    def animate(i):
        print(f"Render step {i}/{n_timesteps}")

        fig.clear()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 1),
                         axes_pad=0.3,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="1.5%",
                         cbar_pad=0.15)

        if FLAGS.mesh_type == "triangle":
            for j, (sim, vel) in enumerate(velocity_result.items()):
                grid[j].triplot(triang, '-', color='k', ms=0.5, lw=2.0)
                handle = grid[j].tripcolor(triang, vel[i], vmax=vmax, vmin=vmin)
                cbar = fig.colorbar(handle, cax=grid.cbar_axes[0])
                cbar.set_label("Velocity (m/s)", rotation=270, labelpad=10)
                fig.suptitle(f"{i}/{n_timesteps}")
                grid[j].set_title(sim, pad=3)
                grid[j].set_xlabel("x (m)")
                grid[j].set_ylabel("y (m)")

        if FLAGS.mesh_type == "quad":
            for j, (sim, vel) in enumerate(velocity_result.items()):
                # Reshape data to fit my lbm grid structure. Current grid: (160 * 40)
                vel_grid = vel[i].reshape(FLAGS.nnode_y, FLAGS.nnode_x)
                x_grid = result["node_coords"][0][:, 0].reshape(FLAGS.nnode_y, FLAGS.nnode_x)
                y_grid = result["node_coords"][0][:, 1].reshape(FLAGS.nnode_y, FLAGS.nnode_x)

                # make the contour plot
                velocity_contour = grid[j].contourf(x_grid, y_grid, vel_grid, 50, cmap='viridis', levels=levels)
                # # Scatter plot for each node type with different color
                # grid[j].scatter(result["node_coords"][0][mask_normal, 0],
                #             result["node_coords"][0][mask_normal, 1], color='blue', s=1, label='Normal nodes')
                # grid[j].scatter(result["node_coords"][0][mask_inflow, 0],
                #             result["node_coords"][0][mask_inflow, 1], color='orange', s=10, label='Inlet nodes')
                # grid[j].scatter(result["node_coords"][0][mask_outflow, 0],
                #             result["node_coords"][0][mask_outflow, 1], color='green', s=10, label='Outlet nodes')
                # grid[j].scatter(result["node_coords"][0][mask_wall, 0],
                #             result["node_coords"][0][mask_wall, 1], color='red', s=3, label='Wall boundary')
                cbar = fig.colorbar(velocity_contour, cax=grid.cbar_axes[0])
                cbar.set_label("Velocity (m/s)", rotation=270, labelpad=10)
                fig.suptitle(f"{i}/{n_timesteps}")
                grid[j].set_aspect('equal')
                grid[j].set_title(sim, pad=3)
                grid[j].set_xlabel("x (m)")
                grid[j].set_ylabel("y (m)")

    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, n_timesteps, FLAGS.step_stride), interval=20)

    ani.save(f'{animation_filename}', dpi=100, fps=10, writer='imagemagick')
    print(f"Animation saved to: {animation_filename}")


def main(_):
    render_gif_animation()


if __name__ == '__main__':
    app.run(main)
