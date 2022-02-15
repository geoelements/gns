import torch
import numpy as np

class TrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self, path, input_length_sequence):
        super().__init__()
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        # TODO (jpv): allow_pickle=True is potential security risk. See docs.
        self._data = [item for _, item in np.load(path, allow_pickle=True).items()]
        
        # length of each trajectory in the dataset
        # excluding the input_length_sequence - 1
        # may (and likely is) variable between data
        self._input_length_sequence = input_length_sequence
        self._ignore_nsteps = input_length_sequence - 1
        self._data_lengths = [x.shape[0] - self._ignore_nsteps for x, _ in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(len(self._data_lengths))]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Select the trajectory immediately before
        # the one that exceeds the idx (i.e., the one in which
        # idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx]
        time_idx = self._ignore_nsteps + (idx - start_of_selected_trajectory)

        # Prepare training data.
        positions = self._data[trajectory_idx][0][time_idx - self._input_length_sequence+1:time_idx+1]
        particle_type = self._data[trajectory_idx][1]
        n_particles_per_example = positions.shape[1]
        label = self._data[trajectory_idx][0][time_idx+1]

        return ((positions, particle_type, n_particles_per_example), label)

def get_data_loader(path, input_length_sequence, batch_size):
    dataset = TrajectoryDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
