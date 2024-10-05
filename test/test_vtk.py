import pytest
import numpy as np
import pickle
import os
import tempfile
import shutil
import pyvista as pv
from gns.train import rendering
from omegaconf import DictConfig

@pytest.fixture
def cfg_vtk():
    """
    Fixture for VTK configuration.

    Returns:
        DictConfig: Configuration dictionary for VTK rendering mode.
    """
    return DictConfig({
        'rendering': {
            'mode': 'vtk'
        }
    })

@pytest.fixture
def temp_dir():
    """
    Fixture for creating and cleaning up a temporary directory.

    Yields:
        str: Path to the temporary directory.
    """
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)

@pytest.fixture
def dummy_pkl_data(temp_dir):
    """
    Fixture for generating dummy pickle data for testing.

    Args:
        temp_dir (str): Path to the temporary directory.

    Returns:
        tuple: Path to the temporary directory and the pickle file name.
    """
    # Define parameters for simulation
    n_timesteps = 2
    n_particles = 3
    dim = 2
    n_init_pos = 2

    # Generate random predictions and ground truth positions
    predictions = np.random.rand(n_timesteps, n_particles, dim)
    ground_truth_positions = np.random.randn(n_timesteps, n_particles, dim)
    loss = (predictions - ground_truth_positions)**2

    # Rollout dictionary to store all relevant information
    dummy_rollout = {
        "initial_positions": np.random.rand(n_init_pos, n_particles, dim),
        "predicted_rollout": predictions,
        "ground_truth_rollout": ground_truth_positions,
        "particle_types": np.full(n_particles, 5)
    }

    # Metadata for the simulation
    metadata = {
        "bounds": [[0.0, 1.0], [0.0, 1.0]]
    }

    dummy_rollout['metadata'] = metadata
    dummy_rollout['loss'] = loss.mean()
    pkl_file_name = "test_input_file.pkl"
    pkl_file_path = os.path.join(temp_dir, pkl_file_name)
    with open(pkl_file_path, "wb") as f:
        pickle.dump(dummy_rollout, f)
    temp_dir = temp_dir + '/'
    pkl_file_name = "test_input_file"

    return temp_dir, pkl_file_name

def n_files(dir, extension):
    """
    Count the number of files with a specific extension in a directory.

    Args:
        dir (str): Directory path.
        extension (str): File extension to count.

    Returns:
        int: Number of files with the specified extension.
    """
    files = os.listdir(dir)
    each_file = []

    for file in files:
        if file.endswith(extension):
            each_file.append(file)

    return len(each_file)

def test_rendering_vtk(dummy_pkl_data, cfg_vtk):
    """
    Test the VTK rendering function.

    Args:
        dummy_pkl_data (tuple): Tuple containing the path to the temporary directory and the pickle file name.
        cfg_vtk (DictConfig): Configuration dictionary for VTK rendering mode.
    """
    input_dir, input_file = dummy_pkl_data
    
    rendering(input_dir, input_file, cfg_vtk)

    # Define paths for the generated VTK files
    vtk_path_gns = os.path.join(input_dir, f"{input_file}_vtk-GNS")
    vtk_path_reality = os.path.join(input_dir, f"{input_file}_vtk-Reality")

    with open(f"{input_dir}{input_file}.pkl", "rb") as file:
        rollout = pickle.load(file)

    # Concatenate initial positions and rollout positions
    positions_gns = np.concatenate(
        [rollout["initial_positions"], rollout["predicted_rollout"]],
        axis=0,
    )
    positions_reality = np.concatenate(
        [rollout["initial_positions"], rollout["ground_truth_rollout"]],
        axis=0,
    )
    
    # Count the number of .vtu and .vtr files in the VTK directories
    n_vtu_files_gns = n_files(vtk_path_gns, 'vtu')
    n_vtu_files_reality = n_files(vtk_path_reality, 'vtu')
    n_vtr_files_gns = n_files(vtk_path_gns, 'vtr')
    n_vtr_files_reality = n_files(vtk_path_reality, 'vtr')

    # Assert that the number of .vtu and .vtr files matches the expected count
    assert n_vtu_files_gns == (positions_gns.shape[0])
    assert n_vtu_files_reality == (positions_reality.shape[0])
    assert n_vtr_files_gns == (positions_gns.shape[0])
    assert n_vtr_files_reality == (positions_reality.shape[0])

    # Verify the contents of the generated VTK files at each time step for GNS
    for time_step in range(positions_gns.shape[0]):
        vtu = os.path.join(vtk_path_gns, f"points{time_step}.vtu")
        vtu_object = pv.read(vtu)
        displacement = vtu_object['displacement']
        particle_type = vtu_object['particle_type']
        color_map = vtu_object['color']

        assert np.all(displacement == np.linalg.norm(positions_gns[0] - positions_gns[time_step], axis=1))
        assert np.all(particle_type == rollout['particle_types'])
        assert np.all(color_map == rollout['particle_types'])

        vtr = os.path.join(vtk_path_gns, f"boundary{time_step}.vtr")
        vtr_object = pv.read(vtr)

        bounds = vtr_object.bounds

        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        assert xmin == rollout['metadata']['bounds'][0][0]
        assert xmax == rollout['metadata']['bounds'][0][1]
        assert ymin == rollout['metadata']['bounds'][1][0]
        assert ymax == rollout['metadata']['bounds'][1][1]

    # Verify the contents of the generated VTK files at each time step for reality
    for time_step in range(positions_reality.shape[0]):
        vtu = os.path.join(vtk_path_reality, f"points{time_step}.vtu")
        vtu_object = pv.read(vtu)
        displacement = vtu_object['displacement']
        particle_type = vtu_object['particle_type']
        color_map = vtu_object['color']

        assert np.all(displacement == np.linalg.norm(positions_reality[0] - positions_reality[time_step], axis=1))
        assert np.all(particle_type == rollout['particle_types'])
        assert np.all(color_map == rollout['particle_types'])

        vtr = os.path.join(vtk_path_reality, f"boundary{time_step}.vtr")
        vtr_object = pv.read(vtr)

        bounds = vtr_object.bounds

        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        assert xmin == rollout['metadata']['bounds'][0][0]
        assert xmax == rollout['metadata']['bounds'][0][1]
        assert ymin == rollout['metadata']['bounds'][1][0]
        assert ymax == rollout['metadata']['bounds'][1][1]