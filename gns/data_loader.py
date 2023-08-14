import torch
import numpy as np

def load_npz_data(path):
    """Load data stored in npz format.
    
    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.

    Args:
        path (str): Path to npz file.
    
    Returns:
        data (list): List of tuples of the form (positions, particle_type).
    """
    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
    return data

class SamplesDataset(torch.utils.data.Dataset):
    """Dataset of samples of trajectories.
    
    Each sample is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    particle_type is an integer.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.

    Attributes:
        _data (list): List of tuples of the form (positions, particle_type).
        _dimension (int): Dimension of the data.
        _input_length_sequence (int): Length of input sequence.
        _data_lengths (list): List of lengths of trajectories in the dataset.
        _length (int): Total number of samples in the dataset.
        _precompute_cumlengths (np.array): Precomputed cumulative lengths of trajectories in the dataset.
    """
    def __init__(self, path, input_length_sequence):
        super().__init__()
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        # TODO: allow_pickle=True is potential security risk. See docs.
        self._data = load_npz_data(path)
        
        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._dimension = self._data[0][0].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _ in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths)+1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):
        """Return length of dataset.
        
        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.
        
        Args:
            idx (int): Index of training example.

        Returns:
            tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).
        """
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
    """Collate function for SamplesDataset.

    Args:
        data (list): List of tuples of the form ((positions, particle_type, n_particles_per_example), label).

    Returns:
        tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).    
    """
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
        torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(), 
        torch.tensor(np.concatenate(particle_type_list)).contiguous(),
        torch.tensor(n_particles_per_example_list).contiguous(),
        ),
        torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous()
        )


class TrajectoriesDataset(torch.utils.data.Dataset):
    """Dataset of trajectories.

    Each trajectory is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    """
    def __init__(self, path):
        super().__init__()
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        # TODO (jpv): allow_pickle=True is potential security risk. See docs.
        self._data = load_npz_data(path)
        self._dimension = self._data[0][0].shape[-1]
        self._length = len(self._data)

    def __len__(self):
        """Return length of dataset.
        
        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.
        
        Args:
            idx (int): Index of training example.
            
        Returns:
            tuple: Tuple of the form (positions, particle_type).
        """
        positions, _particle_type = self._data[idx]
        positions = np.transpose(positions, (1, 0, 2))
        particle_type = np.full(positions.shape[0], _particle_type, dtype=int)
        n_particles_per_example = positions.shape[0]
        return (
            torch.tensor(positions).to(torch.float32).contiguous(), 
            torch.tensor(particle_type).contiguous(), 
            n_particles_per_example
        )



def get_data_loader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    """Returns a data loader for the dataset.
    
    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        
    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = SamplesDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)

def get_data_loader_by_trajectories(path):
    """Returns a data loader for the dataset.
    
    Args:
        path (str): Path to dataset.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = TrajectoriesDataset(path)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)