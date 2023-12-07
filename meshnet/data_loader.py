import torch
import numpy as np
import torch_geometric
from torch_geometric.data import Data

class SamplesDataset(torch.utils.data.Dataset):

    def __init__(self, path, input_length_sequence, dt, fixed_mesh):
        super().__init__()
        # load dataset stored in npz format.
        # data consists of dict with keys:
        # ["pos", "node_type", "velocity", "cells", "pressure"] for all trajectory.
        # whose shapes are (600, 1876, 2), (600, 1876, 1), (600, 1876, 2), (600, 3518, 3), (600, 1876, 1)
        # convert to list of tuples
        self._data = [dict(trj_info.item()) for trj_info in np.load(path, allow_pickle=True).values()]

        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._dimension = self._data[0]["velocity"].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._dt = dt
        self._data_lengths = [x["velocity"].shape[0] - input_length_sequence for x in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths) + 1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

        # Mesh-related data doesn't change with time if `fixed_mesh=True`
        self.fixed_mesh = fixed_mesh

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx-1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)

        # Prepare training data. Assume `input_sequence_length`=1
        if self.fixed_mesh:
            # if simulation is fixed mesh, mesh-related data is the same over the simulation time
            positions = self._data[trajectory_idx]["pos"][0]  # (nnode, dimension)
            cells = self._data[trajectory_idx]["cells"][0]  # (ncells, nnode_per_cell)
            cells = np.transpose(cells, (1, 0))
            node_type = self._data[trajectory_idx]["node_type"][0]  # (nnode, 1)
        else:
            # else, mesh-related data changes over the time
            positions = self._data[trajectory_idx]["pos"][time_idx - 1]  # (nnode, dimension)
            cells = self._data[trajectory_idx]["cells"][time_idx - 1]  # (ncells, nnode_per_cell)
            cells = np.transpose(cells, (1, 0))
            node_type = self._data[trajectory_idx]["node_type"][time_idx - 1]  # (nnode, 1)
        velocity_feature = self._data[trajectory_idx]["velocity"][time_idx - 1]  # (nnode, dimension)
        velocity_target = self._data[trajectory_idx]["velocity"][time_idx]  # (nnode, dimension)
        if "pressure" in self._data[trajectory_idx]:
            pressure = self._data[trajectory_idx]["pressure"][time_idx - 1]  # (nnode, 1)
        time_idx_vector = np.full(velocity_feature.shape[0], time_idx - 1)  # (nnode, )
        time_vector = time_idx_vector * self._dt  # (nnodes, )
        time_vector = np.reshape(time_vector, (time_vector.size, 1))  # (nnodes, 1)

        # aggregate node features
        if "pressure" in self._data[trajectory_idx]:
            node_features = torch.hstack(
                (torch.tensor(node_type).contiguous(),
                 torch.tensor(velocity_feature).to(torch.float32).contiguous(),
                 torch.tensor(pressure).to(torch.float32).contiguous(),
                 torch.tensor(time_vector).to(torch.float32).contiguous())
            )
        else:
            node_features = torch.hstack(
                (torch.tensor(node_type).contiguous(),
                 torch.tensor(velocity_feature).to(torch.float32).contiguous(),
                 torch.tensor(time_vector).to(torch.float32).contiguous())
            )


        # make graph
        graph = Data(x=node_features,
                     face=torch.tensor(cells).type(torch.LongTensor).contiguous(),
                     y=torch.tensor(velocity_target).to(torch.float32).contiguous(),
                     pos=torch.tensor(positions).to(torch.float32).contiguous())

        return graph


class TrajectoriesDataset(torch.utils.data.Dataset):

    def __init__(self, path, fixed_mesh):
        super().__init__()
        # load dataset stored in npz format.
        # data consists of dict with keys:
        # ["pos", "node_type", "velocity", "cells", "pressure"] for all trajectory.
        # whose shapes are (600, 1876, 2), (600, 1876, 1), (600, 1876, 2), (600, 3518, 3), (600, 1876, 1)
        # convert to list of tuples
        self._data = [dict(trj_info.item()) for trj_info in np.load(path, allow_pickle=True).values()]

        # Get necessary information of data
        self._dimension = self._data[0]["velocity"].shape[-1]
        self._length = len(self._data)

        # Mesh-related data doesn't change with time if `fixed_mesh=True`
        self.fixed_mesh = fixed_mesh

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        velocity_feature = self._data[idx]["velocity"]  # (timesteps, nnode, dims)
        if "pressure" in self._data[idx]:
            pressure = self._data[idx]["pressure"]  # (timesteps, nnode, 1)
        ntimesteps = self._data[idx]["velocity"].shape[0]
        if self.fixed_mesh:
            position = self._data[idx]["pos"]  # (1, nnode, dims)
            positions = np.tile(position, (ntimesteps, 1, 1))  # (timesteps, nnode, dims)
            n_node_per_example = position.shape[1]  # (nnode, )
            node_type = self._data[idx]["node_type"]  # (1, nnode, 1)
            node_types = np.tile(node_type, (ntimesteps, 1, 1))  # (timesteps, nnode, 1)
            cell = self._data[idx]["cells"]  # (1, ncell, nnode_per_cell)
            cells = np.tile(cell, (ntimesteps, 1, 1))  # (timesteps, ncell, nnode_per_cell)
        else:
            positions = self._data[idx]["pos"]  # (timesteps, nnode, dims)
            n_node_per_example = positions.shape[1]  # (nnode, )
            node_types = self._data[idx]["node_type"]  # (timesteps, nnode, 1)
            cells = self._data[idx]["cells"]  # (timesteps, ncell, nnode_per_cell)

        if "pressure" in self._data[idx]:
            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(node_types).contiguous(),
                torch.tensor(velocity_feature).to(torch.float32).contiguous(),
                torch.tensor(cells).to(torch.float32).contiguous(),
                torch.tensor(pressure).to(torch.float32).contiguous(),
                torch.tensor(n_node_per_example).contiguous(),
            )
        else:
            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(node_types).contiguous(),
                torch.tensor(velocity_feature).to(torch.float32).contiguous(),
                torch.tensor(cells).to(torch.float32).contiguous(),
                torch.tensor(n_node_per_example).contiguous(),
            )

        return trajectory

def get_data_loader_by_samples(path, input_length_sequence, dt, batch_size, shuffle=True, fixed_mesh=True):
    dataset = SamplesDataset(path, input_length_sequence, dt, fixed_mesh)
    return torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_data_loader_by_trajectories(path, fixed_mesh=True):
    dataset = TrajectoriesDataset(path, fixed_mesh)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)
