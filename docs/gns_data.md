# GNS Data
To train or run GNS, we need two inputs:
* `.npz`: includes the trajectory information of particles. 
* `metadata.json`: defines the configuration of GNS.

## `.npz`
We use the numpy `.npz` format for storing data for training GNS for particulate domain.

`.npz` contains python dictionary describing the trajectory information.
Each trajectory includes the positions of particles over time and 
their material properties (optional). Specifically, this dictionary includes:

```python
simulation_data = {
    "simulation_0": (
        positions,
        particle_types,
        material_properties  # Optional
    )
    "simulation_1": (
        ...
    ),
    
    ...
    
    "simulation_n": (
    ...
    )
}
```

The details are as follows.

### `positions`
`positions` contains a numpy array that describes positions of particles for each 
timesteps with `shape=(ntimestep, nnodes, ndims)`. 

```python
array([[[px0_t0, py0_t0],  # at t=0
        [px1_t0, py1_t0],
        ...,
        [px16_t0, py16_t0],
        [px17_t0, py17_t0]],
       [[px0_t1, py0_t1],  # at t=1
        [px1_t1, py1_t1],
        ...,
        [px16_t1, py16_t1]
        [px17_t1, py17_t1]],
       
       ...

       [[px0_t99, py0_t99],  # at t=99
        [px1_t99, py1_t99],
        ...,
        [px16_t99, py16_t99]
        [px17_t99, py17_t99]]], shape=(100, 18, 2))
```

### `particle_types`
`particle_types` is a numpy array containing integers that represents the material type
for each particle that corresponds to `positions`. The shape=(n_particles,).

By default, stationary particle type is set to `3`. You may assign any other integer
for distinguishing the type of particles. 

For example, the `BarrierInteraction` dataset in [here](https://doi.org/10.17603/ds2-4nqz-s548),
we use `6` to represent sand, and `3` to represent stationary particles for barrier.
```python
array([6,  # particle-0: sand
       6,  # particle-1: sand
       6,  # particle-2: sand
       3,  # particle-3: stationary
       3,  # particle-4: stationary
       ...,
       3,
       3,
       6
       ])
```

### `material_properties` (optional)
`material_properties` is a numpy array that describes the material properties 
for each particle that corresponds to `positions`. The shape=(n_particles, n_material_properties).

This will be appended to the end of the node features.

For example, the `ColumnCollapseFrictional` dataset in [here](https://doi.org/10.17603/ds2-4nqz-s548),
we use the normalized friction angle `tan(phi)` where `phi` is the friction angle of sand material in degrees. 

```python
array([tan(np.deg2rad(30)),  # particle-0
       tan(np.deg2rad(30)),  # particle-1
       tan(np.deg2rad(45)),  # particle-2
       tan(np.deg2rad(45)),  # particle-3
       tan(np.deg2rad(45)),  # particle-4
       ...,
       tan(np.deg2rad(30)),
       tan(np.deg2rad(30)),
       tan(np.deg2rad(30))
       ])
```
> This is just an example. We did not use multiple friction angles in a single trajectory in 
> `ColumnCollapseFrictional` dataset. 


### Save and load
Once the python dictionary is ready, the following lines saves the entire dictionary in a compressed format (`.npz`).

```python
import numpy as np
np.savez_compressed("train.npz", **trajectories)
```

To load data, 
```python
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
```

# `metadata.json`
It defines configuration of GNS.
```json

{
  "bounds": [[0.1, 0.9], [0.1, 0.9]],
  "sequence_length": 320, 
  "default_connectivity_radius": 0.015, 
  "dim": 2, 
  "dt": 0.0025, 
  "vel_mean": [5.123277536458455e-06, -0.0009965205918140803], 
  "vel_std": [0.0021978993231675805, 0.0026653552458701774], 
  "acc_mean": [5.237611158734309e-07, 2.3633027988858656e-07], 
  "acc_std": [0.0002582944917306106, 0.00029554531667679154]
}
```

* `bounds`: boundaries of simulation domain for each dim (e.g., `[[x_min, x_max], [y_min, y_max], [z_min, z_max]]`).
* `sequence_length`: number of timesteps in your trajectory (e.g., `320`).
* `default_connectivity_radius`: a radius that defines graph connectivity from a node (e.g., `0.015`). 
* `dim`: dimensionality of the simulation (e.g., `3`). 
* `dt`: physical time interval between each timestep (e.g., `0.0025`).  
* `vel_mean`: mean velocity over all trajectories and timesteps in `train.npz` (e.g., [5.123277536458455e-06, -0.0009965205918140803]).
* `vel_std`: velocity standard deviation over all trajectories and timesteps in `train.npz` (e.g., [0.0021978993231675805, 0.0026653552458701774]).
* `acc_mean`: mean acceleration over all trajectories and timesteps in `train.npz` (e.g., [5.237611158734309e-07, 2.3633027988858656e-07]). 
* `acc_std`: acceleration standard deviation over all trajectories and timesteps in `train.npz` (e.g., [0.0002582944917306106, 0.00029554531667679154]).

> In the current implementation, `dt` is not used.
> For the training purpose, `sequence_length` can be ignored. 

For reference, the following code computes the mean and std for use in `metadata.json` 

```python
# Assuming trajectories is your dictionary of velocity arrays
trajectories = {
    'trajectory-1': velocity1,
    'trajectory-2': velocity2,
    ...
}

# Concatenate all velocity arrays along the time axis to get the shape (n_trajectories * n_timesteps, n_particles, n_dims)
all_velocities = np.concatenate(list(trajectories.values()), axis=0)  # Shape: (n_trajectories * n_timesteps, n_particles, n_dims)

# Compute mean and std over the concatenated time axis (axis=0)
mean_velocity = np.mean(all_velocities, axis=0)
std_velocity = np.std(all_velocities, axis=0)
```

Note that, for a large dataset, you may need to use the [running sum](https://github.com/geoelements/gns/issues/76#issue-2329501493)
to avoid excessive memory occupation.

If you want to use a different configuration between train and test, 
the `metadata.json` allows the separate designation for `train` and `rollout` (i.e., test).
```json
{
  "train": {
    "bounds": [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]],
    "sequence_length": 350,
    "default_connectivity_radius": 0.029,
    "material_feature_len": 1,
    "dim": 3,
    "dt": 0.0025,
    "nnode_in": 38,
    "nedge_in": 4,
    "nmessage_passing_steps": 10,
    "vel_mean": [5.405872395709705e-06, -0.00047905351802700585, 1.9795009078952637e-06],
    "vel_std": [0.0013014068536713356, 0.0012835209923215106, 0.0012917992322734968],
    "acc_mean": [-9.196989449745803e-10, -2.15021076995823e-08, 9.345642716772633e-10],
    "acc_std": [0.00019060602267917417, 0.00017215552146654462, 0.00019132722713896078],
  },
  "rollout": {
    "bounds": [[0.1, 1.9], [0.1, 0.9], [0.1, 1.9]],  # test in enlarged domain
    "sequence_length": 500,  # rollout for a longer timestep
    "default_connectivity_radius": 0.029,
    "material_feature_len": 1,
    "dim": 3,
    "dt": 0.0025,
    "nnode_in": 38,
    "nedge_in": 4,
    "nmessage_passing_steps": 10,
    "vel_mean": [5.405872395709705e-06, -0.00047905351802700585, 1.9795009078952637e-06],
    "vel_std": [0.0013014068536713356, 0.0012835209923215106, 0.0012917992322734968],
    "acc_mean": [-9.196989449745803e-10, -2.15021076995823e-08, 9.345642716772633e-10],
    "acc_std": [0.00019060602267917417, 0.00017215552146654462, 0.00019132722713896078]
  }
}
```

As you can see above, `metadata.json` supports some optional entries:

* `material_feature_len`: the length of material features. If your `.npz` has `material_properties`,
it should be specified corresponding to the length of `material_properties` for the second axis. 
For example, in `ColumnCollapseFrictional` dataset in [here](https://doi.org/10.17603/ds2-4nqz-s548),
`material_feature_len` should be `1` since it only has the friction angle as a material property. 
* `nnode_in`: the number of input node features.
* `nedge_in`: the number of input edge features.
* `nmessage_passing_steps`: the number of message passing steps.
