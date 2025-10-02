import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import h5py


def load_data(path):
    """Load data stored in npz or h5 format."""
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as data_file:
            if "gns_data" in data_file:
                data = data_file["gns_data"]
            else:
                data = [item for _, item in data_file.items()]
    elif path.endswith(".h5"):
        with h5py.File(path, "r") as data_file:
            data = []
            for key in data_file.keys():
                trajectory = data_file[key]
                positions = trajectory["positions"][()]
                particle_type = trajectory["particle_type"][()]
                material_property = trajectory["material_property"][()]
                data.append((positions, particle_type, material_property))
    else:
        raise ValueError("Unsupported file format. Use .npz or .h5 files.")
    return data


class ParticleDataset(Dataset):
    def __init__(self, file_path, input_sequence_length=6, mode="sample"):
        self.file_path = file_path
        self.input_sequence_length = input_sequence_length
        self.mode = mode
        self.data = load_data(file_path)
        self._preprocess_data()

    def _preprocess_data(self):
        self.dimension = self.data[0][0].shape[-1]
        self.material_property_as_feature = len(self.data[0]) >= 3

        if self.mode == "sample":
            self.data_lengths = [
                x.shape[0] - self.input_sequence_length for x, *_ in self.data
            ]
            self.length = sum(self.data_lengths)
            self.cumulative_lengths = np.cumsum([0] + self.data_lengths)
        else:  # trajectory mode
            self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == "sample":
            return self._get_sample(idx)
        else:  # trajectory mode
            return self._get_trajectory(idx)

    def _get_sample(self, idx):
        trajectory_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        time_idx = (
            idx - self.cumulative_lengths[trajectory_idx] + self.input_sequence_length
        )

        positions = self.data[trajectory_idx][0][
            time_idx - self.input_sequence_length : time_idx
        ]
        positions = np.transpose(positions, (1, 0, 2))
        particle_type = np.full(
            positions.shape[0], self.data[trajectory_idx][1], dtype=int
        )

        n_particles_per_example = positions.shape[0]

        if self.material_property_as_feature:
            material_property = np.full(
                positions.shape[0], self.data[trajectory_idx][2], dtype=float
            )
            features = (
                positions,
                particle_type,
                material_property,
                n_particles_per_example,
            )
        else:
            features = (positions, particle_type, n_particles_per_example)

        label = self.data[trajectory_idx][0][time_idx]

        return features, label

    def _get_trajectory(self, idx):
        if self.material_property_as_feature:
            positions, particle_type, material_property = self.data[idx]
            positions = np.transpose(positions, (1, 0, 2))
            particle_type = np.full(positions.shape[0], particle_type, dtype=int)
            material_property = np.full(
                positions.shape[0], material_property, dtype=float
            )
            n_particles_per_example = positions.shape[0]

            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                torch.tensor(material_property).to(torch.float32).contiguous(),
                n_particles_per_example,
            )
        else:
            positions, particle_type = self.data[idx]
            positions = np.transpose(positions, (1, 0, 2))
            particle_type = np.full(positions.shape[0], particle_type, dtype=int)
            n_particles_per_example = positions.shape[0]

            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                n_particles_per_example,
            )

        return trajectory

    def get_num_features(self):
        """
        Get the number of features in the dataset.

        Returns:
            int: The number of features.
        """
        return len(self.data[0])


def collate_fn_sample(batch):
    """Optimized collation function with pre-allocation and minimal copies."""
    features, labels = zip(*batch)

    # Pre-calculate total particles to avoid reallocation
    total_particles = sum(f[0].shape[0] for f in features)
    batch_size = len(features)
    has_material = len(features[0]) == 4

    # Get dimensions from first sample
    seq_len = features[0][0].shape[1]
    dim = features[0][0].shape[2]

    # Pre-allocate tensors with pinned memory for faster GPU transfer
    positions = torch.empty((total_particles, seq_len, dim),
                           dtype=torch.float32, pin_memory=True)
    particle_types = torch.empty(total_particles,
                                 dtype=torch.long, pin_memory=True)
    n_particles = torch.empty(batch_size,
                             dtype=torch.long, pin_memory=True)

    if has_material:
        materials = torch.empty(total_particles,
                               dtype=torch.float32, pin_memory=True)

    # Fill pre-allocated tensors (single copy from numpy)
    offset = 0
    for i, feature in enumerate(features):
        n_part = feature[0].shape[0]

        # Direct numpy-to-torch copy
        positions[offset:offset+n_part] = torch.from_numpy(feature[0])
        particle_types[offset:offset+n_part] = torch.from_numpy(feature[1])

        if has_material:
            materials[offset:offset+n_part] = torch.from_numpy(feature[2])
            n_particles[i] = feature[3]
        else:
            n_particles[i] = feature[2]

        offset += n_part

    # Build output tuple
    if has_material:
        collated_features = (positions, particle_types, materials, n_particles)
    else:
        collated_features = (positions, particle_types, n_particles)

    # Labels - same optimization
    labels_tensor = torch.empty((total_particles, dim),
                               dtype=torch.float32, pin_memory=True)
    offset = 0
    for label in labels:
        n_part = label.shape[0]
        labels_tensor[offset:offset+n_part] = torch.from_numpy(label)
        offset += n_part

    return collated_features, labels_tensor


def collate_fn_trajectory(batch):
    return batch  # No need for collation as each item is already a full trajectory


def get_data_loader(
    file_path,
    mode="sample",
    input_sequence_length=6,
    batch_size=32,
    shuffle=True,
    use_dist=False,
):
    """
    Get a data loader for the ParticleDataset.

    Args:
        file_path (str): Path to the data file.
        mode (str): 'sample' or 'trajectory' mode.
        input_sequence_length (int): Length of input sequence.
        batch_size (int): Batch size for the data loader.
        shuffle (bool): Whether to shuffle the data.
        use_dist (bool): Whether to use DistributedSampler for distributed training.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    dataset = ParticleDataset(file_path, input_sequence_length, mode)

    if use_dist:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # DistributedSampler handles shuffling
    else:
        sampler = None

    if mode == "sample":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn_sample,
            pin_memory=True,
        )
    else:  # trajectory mode
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn_trajectory,
            pin_memory=True,
        )
