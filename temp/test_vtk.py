import numpy as np
import pickle
import os
from gns.train import rendering
from omegaconf import DictConfig
import pyvista as pv

#configuration dictionary for VTK rendering
def cfg_vtk():
    return DictConfig({
        'rendering': {
            'mode': 'vtk'
    }
    })

#function to count the no of files with a specific extension
#in a directory
def n_files(dir, extension):
    files = os.listdir(dir)
    each_file = []

    for file in files:
        if file.endswith(extension):
            each_file.append(file)

    return len(each_file)

#define parameters for simulation
n_trajectories = 1
n_timesteps = 2
n_particles = 3
dim = 2
n_init_pos = 2

# Generate random predictions and ground truth positions
predictions = np.random.rand(n_timesteps, n_particles, dim)
ground_truth_positions = np.random.randn(n_timesteps, n_particles, dim)
loss = (predictions - ground_truth_positions)**2

# rollout dictionary to store all relevant information
rollout = {
    "initial_positions": np.random.rand(n_init_pos, n_particles, dim),
    "predicted_rollout": predictions, 
    "ground_truth_rollout": ground_truth_positions,
    "particle_types": np.full(n_particles, 5), 
}

# Metadata for the simulation
metadata = {
        "bounds": [[0.0, 1.0], [0.0, 1.0]]
}

# Define output directory and file name
output_dir = './temp'
file_name = 'temp.pkl'
rollout['metadata'] = metadata
rollout['loss'] = loss.mean()
file_path = os.path.join(output_dir, file_name)

# Save the rollout dictionary to a pickle file
with open (file_path, 'wb') as f:
    pickle.dump(rollout, f)

# Update output directory path
output_dir = output_dir + '/'
cfg = cfg_vtk()

# Render the simulation data to VTK format
rendering(output_dir, 'temp', cfg)

# Define paths for the generated VTK files
vtk_path_gns = os.path.join(output_dir, 'temp_vtk-GNS')
vtk_path_reality = os.path.join(output_dir, 'temp_vtk-Reality')

# Count the number of .vtu and .vtr files in the VTK directories
n_vtu_files_gns = n_files(vtk_path_gns, 'vtu')
n_vtu_files_reality = n_files(vtk_path_reality, 'vtu')
n_vtr_files_gns = n_files(vtk_path_gns, 'vtr')
n_vtr_files_reality = n_files(vtk_path_reality, 'vtr')

# Assert that the number of .vtu and .vtr files matches the expected count
assert n_vtu_files_gns == (n_init_pos + n_timesteps)
assert n_vtu_files_reality == (n_init_pos + n_timesteps)
assert n_vtr_files_gns == (n_init_pos + n_timesteps)
assert n_vtr_files_reality == (n_init_pos + n_timesteps)

# Concatenate initial positions and rollout positions
positions_gns = np.concatenate(
                [rollout["initial_positions"], rollout["predicted_rollout"]],
                axis=0,
            )
positions_reality = np.concatenate(
                [rollout["initial_positions"], rollout["ground_truth_rollout"]],
                axis=0,
            )

# Verify the contents of the generated VTK files for GNS
for i in range(n_init_pos + n_trajectories):
    vtu = os.path.join(vtk_path_gns, f"points{i}.vtu")
    vtu_object = pv.read(vtu)
    displacement = vtu_object['displacement']
    particle_type = vtu_object['particle_type']
    color_map = vtu_object['color']

    assert np.all(displacement == np.linalg.norm(positions_gns[0] - positions_gns[i], axis=1))
    assert np.all(particle_type == rollout['particle_types'])
    assert np.all(color_map == rollout['particle_types'])

    vtr = os.path.join(vtk_path_gns, f"boundary{i}.vtr")
    vtr_object = pv.read(vtr)

    bounds = vtr_object.bounds

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    assert xmin == metadata['bounds'][0][0]
    assert xmax == metadata['bounds'][0][1]
    assert ymin == metadata['bounds'][1][0]
    assert ymax == metadata['bounds'][1][1]

# Verify the contents of the generated VTK files for reality
for j in range(n_init_pos + n_trajectories):
    vtu = os.path.join(vtk_path_reality, f"points{j}.vtu")
    vtu_object = pv.read(vtu)
    displacement = vtu_object['displacement']
    particle_type = vtu_object['particle_type']
    color_map = vtu_object['color']

    assert np.all(displacement == np.linalg.norm(positions_reality[0] - positions_reality[j], axis=1))
    assert np.all(particle_type == rollout['particle_types'])
    assert np.all(color_map == rollout['particle_types'])

    vtr = os.path.join(vtk_path_reality, f"boundary{j}.vtr")
    vtr_object = pv.read(vtr)

    bounds = vtr_object.bounds

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    assert xmin == metadata['bounds'][0][0]
    assert xmax == metadata['bounds'][0][1]
    assert ymin == metadata['bounds'][1][0]
    assert ymax == metadata['bounds'][1][1]