import pytest
import numpy as np
import os
import tempfile
import shutil
import sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from gns.data_loader import SamplesDataset, TrajectoriesDataset


@pytest.fixture
def temp_dir():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


@pytest.fixture
def dummy_data(temp_dir):
    # Create a dummy dataset
    dummy_data = [(np.random.rand(10, 3, 2), i % 3) for i in range(5)]

    # Create structured array to hold the data
    structured_data = np.empty(len(dummy_data), dtype=object)
    for i, item in enumerate(dummy_data):
        structured_data[i] = item

    # Save the data
    data_path = os.path.join(temp_dir, "data.npz")
    np.savez(data_path, gns_data=structured_data)

    return data_path, dummy_data


def test_samples_dataset(dummy_data):
    data_path, _ = dummy_data
    input_length_sequence = 5
    dataset = SamplesDataset(data_path, input_length_sequence)
    assert (
        len(dataset) == 25
    )  # 5 (trajectories) * (10 (positions) - 5 (input_length_sequence))
    for i in range(len(dataset)):
        ((positions, particle_type, n_particles_per_example), label) = dataset[i]
        assert positions.shape == (3, input_length_sequence, 2)  # Check positions shape
        assert particle_type.shape == (3,)  # Check particle_type shape
        assert n_particles_per_example == 3  # Check number of particles per example
        assert label.shape == (3, 2)  # Check label shape


def test_trajectories_dataset(dummy_data):
    data_path, _ = dummy_data
    dataset = TrajectoriesDataset(data_path)
    assert len(dataset) == 5  # We have 5 trajectories
    for i in range(len(dataset)):
        (positions, particle_type, n_particles_per_example) = dataset[i]
        assert positions.shape == (3, 10, 2)  # Check positions shape
        assert particle_type.shape == (3,)  # Check particle_type shape
        assert n_particles_per_example == 3  # Check number of particles per example
