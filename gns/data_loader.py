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
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._input_length_sequence = input_length_sequence
        self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _ in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths)+1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

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

        # Prepare training data.
        positions = self._data[trajectory_idx][0][time_idx - self._input_length_sequence:time_idx]
        positions = np.transpose(positions, (1, 0, 2)) # nparticles, input_sequence_length, dimension
        particle_type = np.full(positions.shape[0], self._data[trajectory_idx][1], dtype=int)
        n_particles_per_example = positions.shape[0]
        label = self._data[trajectory_idx][0][time_idx]

        return ((positions, particle_type, n_particles_per_example), label)

def collate_fn(data):
    position_list = []
    particle_type_list = []
    n_particles_per_example_list = []
    label_list = []

    for ((positions, particle_type, n_particles_per_example), label) in data:
        position_list.append(positions)
        particle_type_list.append(particle_type)
        n_particles_per_example_list.append(n_particles_per_example)
        label_list.append(label)

    return ((
        torch.tensor(np.vstack(position_list)).contiguous(), 
        torch.tensor(np.concatenate(particle_type_list)).contiguous(),
        torch.tensor(n_particles_per_example_list).contiguous(),
        ),
        torch.tensor(np.vstack(label_list)).contiguous()
        )

def get_data_loader(path, input_length_sequence, batch_size, shuffle=True):
    dataset = TrajectoryDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)
