import pickle
import numpy as np
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
import imageio
from matplotlib import animation
import io
from utils import NodeType


MARKER_SIZE = 10
COLORMAP_TYPE = "viridis"
NAN_COLOR = "black"


def get_pkl(path):

    with open(path, 'rb') as f:
        result = pickle.load(f)
    vel_true = np.concatenate((result["initial_velocities"], result["ground_truth_rollout"]))
    vel_pred = np.concatenate((result["initial_velocities"], result["predicted_rollout"]))
    node_coords = result["node_coords"]
    node_type = result["node_types"]

    return node_coords, node_type, vel_true, vel_pred,

def get_npz(path):
    result = [dict(trj_info.item()) for trj_info in np.load(path, allow_pickle=True).values()]
    if len(result) > 1:
        raise NotImplementedError("Expected to contain only one trajectory in npz")
    node_coords = result[0]["pos"]
    vel = result[0]["velocity"]
    node_type = result[0]["node_type"]

    return node_coords, node_type, vel


class VisMeshNet:
    def __init__(
            self,
            mesh_type,
            node_coords,
            node_type,
            vel_true,
            vel_pred=None,
            quad_grid_config=None
    ):

        self.mesh_type = mesh_type
        self.node_coords = node_coords
        self.node_type = node_type
        self.vel_true = vel_true
        self.vel_pred = vel_pred

        # Compute velocity magnitude
        self.vel_mag_true = np.linalg.norm(vel_true, axis=-1)
        if vel_pred is not None:
            self.vel_mag_pred = np.linalg.norm(vel_pred, axis=-1)

        # Create masks for each node type
        self.mask_normal = np.squeeze(node_type == NodeType.NORMAL)
        self.mask_outflow = np.squeeze(node_type == NodeType.OUTFLOW)
        self.mask_inflow = np.squeeze(node_type == NodeType.INFLOW)
        self.mask_wall = np.squeeze(node_type == NodeType.WALL_BOUNDARY)

        # Necessary variables
        self.ntimesteps = len(vel_true)
        self.lx = quad_grid_config[0]
        self.ly = quad_grid_config[1]

        # Color map
        # Choose a colormap and modify it to display NaN values in a specific color
        self.cmap = plt.get_cmap(COLORMAP_TYPE).copy()  # Make a copy of the colormap
        self.cmap.set_bad(color=NAN_COLOR)  # Set the color for NaN values

    def preprocess(self):
        return None

    def plot_field(self, vis_target, timestep):

        # Select velocity magnitude data to plot
        if vis_target == "vel_mag_true":
            vel_mag = self.vel_mag_true
        elif vis_target == "vel_mag_pred":
            vel_mag = self.vel_mag_pred
        else:
            raise ValueError("`vis_target` is expected to get `vel_true` or `veL_pred`")

        # Get the max and min velocity magnitude values
        vmin = vel_mag[:, :].min()
        vmax = vel_mag[:, :].max()

        # Set contour levels between min and max values of magnitudes
        levels = np.linspace(vmin, vmax, 100)

        # Init figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)

        if self.mesh_type == "triangle":
            raise NotImplementedError

        elif self.mesh_type == "quad":
            # Reshape data
            vel_grid = vel_mag[timestep].reshape(self.ly, self.lx)
            x_grid = self.node_coords[0][:, 0].reshape(self.ly, self.lx)
            y_grid = self.node_coords[0][:, 1].reshape(self.ly, self.lx)

            # Set obstacle location to NaN
            grid_mask_wall = self.mask_wall[0].reshape(self.ly, self.lx)
            vel_grid[grid_mask_wall] = np.nan

            # make the velocity field plot
            extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
            velocity_field = ax.imshow(vel_grid, cmap=self.cmap, extent=extent)
            cbar = fig.colorbar(velocity_field, cax=cax, orientation='vertical')
            # velocity_contour = ax.contourf(x_grid, y_grid, vel_grid, 50, cmap=self.cmap, levels=levels)
            # cbar = fig.colorbar(velocity_contour, cax=cax, orientation='vertical')

            # Format the color bar labels in scientific notation
            cbar.formatter = FuncFormatter(lambda x, _: f'{x:.1e}')
            cbar.update_ticks()
            cbar.set_label("Velocity (m/s)", rotation=270, labelpad=10)
            ax.set_aspect('equal')
            ax.set_title(f"{timestep}/{self.ntimesteps}")
            ax.set_xlabel("lx")
            ax.set_ylabel("ly")
        else:
            raise ValueError

        plt.tight_layout()
        return fig

    def plot_field_compare(self, timestep):

        vel_mag_set = {
            "Ground truth": self.vel_mag_true,
            "MeshNet": self.vel_mag_pred
        }

        # Get the max and min velocity magnitude values
        vmin = np.concatenate((self.vel_mag_true, self.vel_mag_pred)).min()
        vmax = np.concatenate((self.vel_mag_true, self.vel_mag_pred)).max()

        # Setting contour levels between min and max values of magnitudes
        levels = np.linspace(vmin, vmax, 100)

        fig = plt.figure(figsize=(9, 4))
        grid = ImageGrid(
            fig, 111,
            nrows_ncols=(1, 2),
            axes_pad=0.3,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.15)

        if self.mesh_type == "triangle":
            raise NotImplementedError

        elif self.mesh_type == "quad":
            # Reshape node coord to grid
            x_grid = self.node_coords[0][:, 0].reshape(self.ly, self.lx)
            y_grid = self.node_coords[0][:, 1].reshape(self.ly, self.lx)

            # Get obstacle mask for velocity grid
            grid_mask_wall = self.mask_wall[0].reshape(self.ly, self.lx)

            for i, (sim, vel_mag) in enumerate(vel_mag_set.items()):
                # Reshape vel data to grid data
                vel_grid = vel_mag[timestep].reshape(self.ly, self.lx)
                # Set obstacle location to NaN
                vel_grid[grid_mask_wall] = np.nan

                # make the velocity field plot
                extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
                velocity_field = grid[i].imshow(vel_grid, cmap=self.cmap, extent=extent)
                cbar = fig.colorbar(velocity_field, cax=grid.cbar_axes[0], orientation='vertical')

                # Format the color bar labels in scientific notation
                cbar.formatter = FuncFormatter(lambda x, _: f'{x:.1e}')
                cbar.update_ticks()
                cbar.set_label("Velocity (m/s)", rotation=270, labelpad=10)
                fig.suptitle(f"{timestep}/{self.ntimesteps}")
                grid[i].set_aspect('equal')
                grid[i].set_title(sim, pad=3)
                grid[i].set_xlabel("x (m)")
                grid[i].set_ylabel("y (m)")

        else:
            raise ValueError

        # plt.tight_layout()
        return fig

    def animate(self, vis_target, step_stride=5, fps=10):
        """
        Args:
            vis_target (str): "compare" or "vel_mag_true" or "vel_mag_pred"
        Returns: gif object
        """

        print("Start creating animation...")
        # List to hold in-memory frames
        frames = []

        # Generate frames for each timestep and store in memory
        print("Writing figs to buffer...")
        for timestep in tqdm(np.arange(0, self.ntimesteps, step_stride)):
            if vis_target == "vel_mag_true" or vis_target == "vel_mag_pred":
                fig = self.plot_field(vis_target, timestep)
            elif vis_target == "compare":
                fig = self.plot_field_compare(timestep)
            else:
                raise ValueError

            # Convert the plot to an image (numpy array)
            fig.canvas.draw()  # Converts fig object to np.array representing RGB image
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # capture the image data
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frames.append(image)
            plt.close(fig)  # Close the figure to free memory

        # Create a bytes buffer to hold the GIF data
        gif_buffer = io.BytesIO()

        # Compile the in-memory frames into a GIF and store in the bytes buffer
        with imageio.get_writer(gif_buffer, mode='I', fps=fps, format='GIF') as writer:
            for frame in tqdm(frames):
                writer.append_data(frame)

        # Seek to the start of the GIF data in the buffer
        gif_buffer.seek(0)

        return gif_buffer

    def save_gif(self, gif_buffer, output_path):
        print(f"Saving animation")
        # Ensure the buffer's read pointer is at the start
        gif_buffer.seek(0)

        with open(output_path, "wb") as f:
            f.write(gif_buffer.read())

        print(f"Animation saved to {output_path}")

