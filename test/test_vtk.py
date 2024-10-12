import pytest
import numpy as np
import pickle
import os
import tempfile
import shutil
import glob
import pyvista as pv
import logging
from gns.train import rendering
from omegaconf import DictConfig
from typing import Tuple
from utils.count_n_files import n_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def cfg_vtk() -> DictConfig:
    """
    Fixture for VTK configuration.

    Returns:
        DictConfig: Configuration dictionary for VTK rendering mode.
    """
    logger.info("Setting up VTK configuration.")
    return DictConfig({"rendering": {"mode": "vtk"}})


@pytest.fixture
def dummy_pkl_data() -> dict:
    """
    Fixture for generating dummy pickle data for testing.

    Returns:
        Dict: A dictionary containing dummy rollout data.
    """
    n_timesteps = 2
    n_particles = 3
    dim = 2
    n_init_pos = 2

    try:
        # Generate random predictions and ground truth positions
        predictions = np.random.rand(n_timesteps, n_particles, dim)
        ground_truth_positions = np.random.randn(n_timesteps, n_particles, dim)
        loss = (predictions - ground_truth_positions) ** 2

        # Rollout dictionary to store all relevant information
        dummy_rollout = {
            "initial_positions": np.random.rand(n_init_pos, n_particles, dim),
            "predicted_rollout": predictions,
            "ground_truth_rollout": ground_truth_positions,
            "particle_types": np.full(n_particles, 5),
            "metadata": {"bounds": [[0.0, 1.0], [0.0, 1.0]]},
            # MSE loss between predictions and ground truth positions
            "loss": loss.mean(),
        }
        logger.info(
            f"Generated dummy data: {n_particles} particles over {n_timesteps} time steps in {dim} dimensions, "
            f"with {n_init_pos} initial positions"
        )

    except Exception as e:
        logger.error(f"Failed to generate dummy pickle data: {e}")
        raise

    return dummy_rollout


@pytest.fixture
def temp_dir_with_file(dummy_pkl_data: dict) -> Tuple[str, str]:
    """
    Fixture to create a temporary directory and a pickle file containing the
    provided dummy data for testing purposes.

    Args:
        dummy_pkl_data (dict): A dictionary containing the dummy data to be
        serialized and stored in a temporary pickle file.

    Yields:
        Tuple[str, str]: A tuple containing:
            - The path to the temporary directory where the pickle file is stored.
            - The base name of the pickle file (without the '.pkl' extension).
    """
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")

    try:
        with tempfile.NamedTemporaryFile(
            dir=temp_dir, suffix=".pkl", delete=False
        ) as temp_file:
            pkl_file_path = temp_file.name

            with open(pkl_file_path, "wb") as f:
                pickle.dump(dummy_pkl_data, f)

            # get the base file name without any extension
            file_name = os.path.splitext(os.path.basename(pkl_file_path))[0]
            logger.info(f"created temporary '.pkl' file: {file_name}")
            temp_dir += "/"

            yield temp_dir, file_name

    except Exception as e:
        logger.error(
            f"Failed to create a Temporary file for the input rollout data: {e}"
        )

    finally:
        shutil.rmtree(temp_dir)


def verify_vtk_files(rollout_data: dict, label: str, temp_dir: str) -> None:
    """
    Verify the integrity of VTK files against expected data.

    This function checks VTK files (VTU and VTR) for specific properties, ensuring that the
    displacement, particle types, color maps, and bounds match the expected values in the
    provided rollout data.

    Args:
        rollout_data (dict): A dictionary containing
            - 'inital positions'
            - 'predicted_rollout'
            - 'ground_truth_rollout'
            - 'particle_types'
            - 'metadata'
        label (str): The label for the specific rollout data to verify against.
        temp_dir (str): The temporary directory where the VTK files are stored.

    Raises:
        AssertionError: If any of the checks on displacement, particle types, color maps,
        or bounds fail.
        Exception: If there is an error reading or processing the VTK files.
    """
    VTU_PREFIX = "points"
    VTR_PREFIX = "boundary"
    VTU_EXTENSION = "vtu"
    VTR_EXTENSION = "vtr"

    positions = np.concatenate(
        [rollout_data["initial_positions"], rollout_data[label]],
        axis=0,
    )

    for time_step in range(positions.shape[0]):
        try:
            vtu_path = os.path.join(
                temp_dir, f"{VTU_PREFIX}{time_step}.{VTU_EXTENSION}"
            )
            logger.info(f"Reading VTU file: {vtu_path}")
            vtu_object = pv.read(vtu_path)

            displacement = vtu_object["displacement"]
            particle_type = vtu_object["particle_type"]
            color_map = vtu_object["color"]

            assert np.all(
                displacement
                == np.linalg.norm(positions[0] - positions[time_step], axis=1)
            ), (
                f"Displacement mismatch for timestep {time_step}: "
                f"expected {np.linalg.norm(positions[0] - positions[time_step], axis=1)}, "
                f"got {displacement}"
            )
            assert np.all(particle_type == rollout_data["particle_types"]), (
                f"Particle type mismatch for timestep {time_step}: "
                f"expected {rollout_data['particle_types']}, got {particle_type}"
            )
            assert np.all(color_map == rollout_data["particle_types"]), (
                f"Color map mismatch for timestep {time_step}: "
                f"expected {rollout_data['particle_types']}, got {color_map}"
            )

        except AssertionError as e:
            logger.error(
                f"Assertion failed while verifying {VTU_PREFIX}{time_step}.{VTU_EXTENSION}: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Error reading {VTU_PREFIX}{time_step}.{VTU_EXTENSION}: {e}")
            raise

        try:
            vtr_path = os.path.join(
                temp_dir, f"{VTR_PREFIX}{time_step}.{VTR_EXTENSION}"
            )
            logger.info(f"Reading VTR file: {vtr_path}")
            vtr_object = pv.read(vtr_path)

            bounds = vtr_object.bounds
            xmin, xmax, ymin, ymax, zmin, zmax = bounds

            assert xmin == rollout_data["metadata"]["bounds"][0][0], (
                f"Xmin mismatch for timestep {time_step}: "
                f"expected {rollout_data['metadata']['bounds'][0][0]}, got {xmin}"
            )
            assert xmax == rollout_data["metadata"]["bounds"][0][1], (
                f"Xmax mismatch for timestep {time_step}: "
                f"expected {rollout_data['metadata']['bounds'][0][1]}, got {xmax}"
            )
            assert ymin == rollout_data["metadata"]["bounds"][1][0], (
                f"Ymin mismatch for timestep {time_step}: "
                f"expected {rollout_data['metadata']['bounds'][1][0]}, got {ymin}"
            )
            assert ymax == rollout_data["metadata"]["bounds"][1][1], (
                f"Ymax mismatch for timestep {time_step}: "
                f"expected {rollout_data['metadata']['bounds'][1][1]}, got {ymax}"
            )

        except AssertionError as e:
            logger.error(
                f"Assertion failed while verifying {VTR_PREFIX}{time_step}.{VTR_EXTENSION}: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Error reading {VTR_PREFIX}{time_step}.{VTR_EXTENSION}: {e}")
            raise

    logger.info("VTK file verification complete.")


def test_rendering_vtk(
    temp_dir_with_file: Tuple[str, str], cfg_vtk: DictConfig
) -> None:
    """
    Test the VTK rendering function.

    Args:
        temp_dir_with_file (Tuple[str, str]): Tuple containing the path to the temporary directory and the pickle file name.
        cfg_vtk (DictConfig): Configuration dictionary for VTK rendering mode.
    """
    input_dir, input_file = temp_dir_with_file

    logger.info(f"Starting rendering test: {input_file}")

    try:
        rendering(input_dir, input_file, cfg_vtk)
    except Exception as e:
        logger.error("Failed to render with dummy rollout data")
        raise

    try:
        # Define paths for the generated VTK files
        VTK_GNS_SUFFIX = "_vtk-GNS"
        VTK_REALITY_SUFFIX = "_vtk-Reality"

        vtk_path_gns = os.path.join(input_dir, f"{input_file}{VTK_GNS_SUFFIX}")
        vtk_path_reality = os.path.join(input_dir, f"{input_file}{VTK_REALITY_SUFFIX}")

        logger.info("Loading rollout data from pickle file.")
        with open(os.path.join(input_dir, f"{input_file}.pkl"), "rb") as file:
            rollout = pickle.load(file)

        # Count the number of .vtu and .vtr files in the VTK directories
        n_vtu_files_gns = n_files(vtk_path_gns, "vtu")
        n_vtu_files_reality = n_files(vtk_path_reality, "vtu")
        n_vtr_files_gns = n_files(vtk_path_gns, "vtr")
        n_vtr_files_reality = n_files(vtk_path_reality, "vtr")

        expected_n_files = (
            rollout["initial_positions"].shape[0]
            + rollout["predicted_rollout"].shape[0]
        )
        logger.info(f"Expected number of files: {expected_n_files}.")

        # Assert that the number of .vtu and .vtr files matches the expected count
        assert (
            n_vtu_files_gns == expected_n_files
        ), f"Expected {expected_n_files} VTU files in GNS path, got {n_vtu_files_gns}"
        assert (
            n_vtu_files_reality == expected_n_files
        ), f"Expected {expected_n_files} VTU files in Reality path, got {n_vtu_files_reality}"
        assert (
            n_vtr_files_gns == expected_n_files
        ), f"Expected {expected_n_files} VTR files in GNS path, got {n_vtr_files_gns}"
        assert (
            n_vtr_files_reality == expected_n_files
        ), f"Expected {expected_n_files} VTR files in Reality path, got {n_vtr_files_reality}"

        logger.info("Verifying VTK files for predicted rollout.")
        verify_vtk_files(rollout, "predicted_rollout", vtk_path_gns)
        logger.info("Verifying VTK files for ground truth rollout.")
        verify_vtk_files(rollout, "ground_truth_rollout", vtk_path_reality)

        logger.info("Rendering test completed successfully.")

    except Exception as e:
        logger.error(f"Rendering test failed: {e}")
