import torch
import numpy as np

from train import prepare_input_data

class TrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self, path, input_length_sequence):
        super().__init__():
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        self._data = list(np.load(path).items())
        
        # length of each trajectory in the dataset
        # excluding the input_length_sequence - 1
        # may (and likely is) variable between data
        self._input_length_sequence = input_length_sequence
        self._ignore_nsteps = input_length_sequence - 1
        self._data_lengths = [x.shape[0] - self._ignore_nsteps for x, _ in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(len(self._length))]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Select the trajectory immediately before
        # the one that exceeds the idx (i.e., the one in which
        # idx resides).
        trajectory_idx = np.argwhere((self._precompute_cumlengths > idx))[0] - 1

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx]
        time_idx = self._ignore_nsteps + (idx - start_of_selected_trajectory)

        # Prepare training data.
        trajectory = self._data[trajectory_idx][0][time_idx - self._input_length_sequence+1:time_idx+1]
        particle_type = self._data[trajectory_idx][1]
        label = self._data[trajectory_idx][0][time_idx+1]

        return ((trajectory, particle_type), label)
