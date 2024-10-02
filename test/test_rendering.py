import pytest
import os
import numpy as np
import tempfile
import shutil
import pickle
from omegaconf import DictConfig
from gns.train import rendering
import sys

@pytest.fixture
def cfg_gif():
    return DictConfig({
        'rendering': {
            'mode': 'gif',
        },
        'gif': {
            'step_stride': 1,
            'change_yz': False,
        }
    })

@pytest.fixture
def cfg_vtk():
    return DictConfig({
        'rendering': {
            'mode': 'vtk',
        },
    })


@pytest.fixture
def temp_dir():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


@pytest.fixture
def dummy_pkl_data(temp_dir):
    n_trajectories = 1
    n_timesteps = 20
    n_particles = 3
    dim = 2

    dummy_metadata = {
        "bounds": [[0.0, 1.0], [0.0, 1.0]],  
        "sequence_length": 100, 
        "default_connectivity_radius": 0.01,
        "dim": 2,
        "dt": 0.001,
        "vel_mean": [0.0, 0.0],
        "vel_std": [0.001, 0.001],
        "acc_mean": [0.0, 0.0],
        "acc_std": [0.0001, 0.0001]
    }

    predictions = np.random.rand(n_timesteps, n_particles, dim)
    ground_truth_positions = np.random.randn(n_timesteps, n_particles, dim)
    loss = (predictions - ground_truth_positions)**2

    dummy_rollout = {
        "initial_positions": np.random.rand(7, n_particles, dim),
        "predicted_rollout": predictions, 
        "ground_truth_rollout": ground_truth_positions,
        "particle_types": np.full(n_particles, n_trajectories%3), 
        "material_property": np.full(n_particles, np.random.rand())
    }
    dummy_rollout["metadata"] = dummy_metadata
    dummy_rollout["loss"] = loss.mean()
    pkl_file_name = "test_input_file.pkl"
    pkl_file_path = os.path.join(temp_dir, pkl_file_name)
    with open(pkl_file_path, "wb") as f:
        pickle.dump(dummy_rollout, f)

    return temp_dir, pkl_file_name


def test_rendering_gif(dummy_pkl_data, cfg_gif):
    input_dir, input_file = dummy_pkl_data
    input_dir += '/'

    rendering(input_dir, "test_input_file", cfg_gif)

    gif_file = os.path.join(input_dir, f"{input_file.replace('.pkl', '')}.gif")
    assert os.path.exists(gif_file), f"GIF file was not created at {gif_file}"


def test_rendering_vtk(dummy_pkl_data, cfg_vtk):
    input_dir, input_file = dummy_pkl_data
    input_dir += '/'

    rendering(input_dir, "test_input_file", cfg_vtk)

    vtk_dir_reality = os.path.join(input_dir, f"{input_file.replace('.pkl', '')}_vtk-Reality")
    vtk_dir_gns = os.path.join(input_dir, f"{input_file.replace('.pkl', '')}_vtk-GNS")

    assert os.path.exists(vtk_dir_reality), f"VTK Reality directory was not created at {vtk_dir_reality}"
    assert os.path.exists(vtk_dir_gns), f"VTK GNS directory was not created at {vtk_dir_gns}"