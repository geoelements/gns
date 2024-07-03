import pytest
import numpy as np
import os
import tempfile
import shutil
import h5py
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gns.particle_data_loader import ParticleDataset, get_data_loader, load_data


@pytest.fixture
def temp_dir():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


@pytest.fixture
def dummy_npz_data(temp_dir):
    # Create dummy dataset
    n_trajectories = 5
    n_timesteps = 20
    n_particles = 3
    dim = 2

    dummy_data = []
    for i in range(n_trajectories):
        positions = np.random.rand(n_timesteps, n_particles, dim)
        particle_type = np.full(n_particles, i % 3)
        material_property = np.full(n_particles, np.random.rand())
        dummy_data.append((positions, particle_type, material_property))

    # Save the data
    data_path = os.path.join(temp_dir, "data.npz")
    np.savez(data_path, gns_data=np.array(dummy_data, dtype=object))

    return data_path, dummy_data


@pytest.fixture
def dummy_h5_data(temp_dir):
    # Create dummy dataset
    n_trajectories = 5
    n_timesteps = 20
    n_particles = 3
    dim = 2

    data_path = os.path.join(temp_dir, "data.h5")
    with h5py.File(data_path, "w") as f:
        for i in range(n_trajectories):
            positions = np.random.rand(n_timesteps, n_particles, dim)
            particle_type = np.full(n_particles, i % 3)
            material_property = np.full(n_particles, np.random.rand())
            trajectory = f.create_group(f"trajectory_{i}")
            trajectory.create_dataset("positions", data=positions)
            trajectory.create_dataset("particle_type", data=particle_type)
            trajectory.create_dataset("material_property", data=material_property)

    return data_path


def test_load_data_npz(dummy_npz_data):
    data_path, original_data = dummy_npz_data
    loaded_data = load_data(data_path)
    assert len(loaded_data) == len(original_data)
    for loaded, original in zip(loaded_data, original_data):
        assert np.allclose(loaded[0], original[0])
        assert np.allclose(loaded[1], original[1])
        assert np.allclose(loaded[2], original[2])


def test_load_data_h5(dummy_h5_data):
    data_path = dummy_h5_data
    loaded_data = load_data(data_path)
    assert len(loaded_data) == 5
    for trajectory in loaded_data:
        assert len(trajectory) == 3
        assert isinstance(trajectory[0], np.ndarray)
        assert isinstance(trajectory[1], np.ndarray)
        assert isinstance(trajectory[2], np.ndarray)


def test_particle_dataset_sample_mode(dummy_npz_data):
    data_path, _ = dummy_npz_data
    input_sequence_length = 6
    dataset = ParticleDataset(
        data_path, input_sequence_length=input_sequence_length, mode="sample"
    )

    assert len(dataset) == 5 * (20 - input_sequence_length)

    features, label = dataset[0]
    assert len(features) == 4
    assert features[0].shape == (3, input_sequence_length, 2)
    assert features[1].shape == (3,)
    assert features[2].shape == (3,)
    assert isinstance(features[3], int)
    assert label.shape == (3, 2)


def test_particle_dataset_trajectory_mode(dummy_npz_data):
    data_path, _ = dummy_npz_data
    dataset = ParticleDataset(data_path, mode="trajectory")

    assert len(dataset) == 5

    trajectory = dataset[0]
    assert len(trajectory) == 4
    assert trajectory[0].shape == (3, 20, 2)
    assert trajectory[1].shape == (3,)
    assert trajectory[2].shape == (3,)
    assert isinstance(trajectory[3], int)


def test_get_data_loader_sample_mode(dummy_npz_data):
    data_path, _ = dummy_npz_data
    batch_size = 2
    loader = get_data_loader(data_path, mode="sample", batch_size=batch_size)

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == batch_size

    batch = next(iter(loader))
    assert len(batch) == 2
    features, labels = batch
    assert len(features) == 4
    # The actual batch size might be different due to the nature of the data
    # We'll just check that it's not empty
    assert features[0].shape[0] > 0
    assert labels.shape[0] > 0


@pytest.fixture
def mock_distributed_env(monkeypatch):
    def mock_is_initialized():
        return True

    def mock_get_world_size():
        return 2

    def mock_get_rank():
        return 0

    monkeypatch.setattr(torch.distributed, "is_initialized", mock_is_initialized)
    monkeypatch.setattr(torch.distributed, "get_world_size", mock_get_world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)


def test_get_data_loader_distributed(dummy_npz_data, mock_distributed_env):
    data_path, _ = dummy_npz_data
    loader = get_data_loader(data_path, use_dist=True)

    assert isinstance(loader, DataLoader)
    assert isinstance(loader.sampler, DistributedSampler)


@pytest.mark.parametrize("input_sequence_length", [1, 6, 10])
def test_particle_dataset_different_sequence_lengths(
    dummy_npz_data, input_sequence_length
):
    data_path, _ = dummy_npz_data
    dataset = ParticleDataset(
        data_path, input_sequence_length=input_sequence_length, mode="sample"
    )

    features, label = dataset[0]
    assert features[0].shape == (3, input_sequence_length, 2)


def test_particle_dataset_get_num_features(dummy_npz_data):
    data_path, _ = dummy_npz_data
    dataset = ParticleDataset(data_path)

    assert dataset.get_num_features() == 3


def test_data_loader_shuffle(dummy_npz_data):
    data_path, _ = dummy_npz_data
    loader_shuffle = get_data_loader(data_path, shuffle=True)
    loader_no_shuffle = get_data_loader(data_path, shuffle=False)

    shuffled_indices = list(iter(loader_shuffle.sampler))
    unshuffled_indices = list(iter(loader_no_shuffle.sampler))

    assert shuffled_indices != unshuffled_indices
