# Graph Network Simulator (GNS) and MeshNet

[![DOI](https://zenodo.org/badge/427487727.svg)](https://zenodo.org/badge/latestdoi/427487727)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/geoelements/gns/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/geoelements/gns/tree/main)
[![Docker](https://quay.io/repository/geoelements/gns/status "Docker Repository on Quay")](https://quay.io/repository/geoelements/gns)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/geoelements/gns/main/license.md)

> Krishna Kumar, The University of Texas at Austin.
> Joseph Vantassel, Texas Advanced Computing Center, UT Austin.
> Yongjin Choi, The University of Texas at Austin.

Graph Network-based Simulator (GNS) is a generalizable, efficient, and accurate machine learning (ML)-based surrogate simulator for particulate and fluid systems using Graph Neural Networks (GNNs). GNS code is a viable surrogate for numerical methods such as Material Point Method, Smooth Particle Hydrodynamics and Computational Fluid dynamics. GNS exploits distributed data parallelism to achieve fast multi-GPU training. The GNS code can handle complex boundary conditions and multi-material interactions.

MeshNet is a scalable surrogate simulator for any mesh-based models like Finite Element Analysis (FEA), Computational Fluid Dynammics (CFD), and Finite Difference Methods (FDM). 

## Run GNS/MeshNet

> Training GNS/MeshNet on simulation data
```shell
# For particulate domain,
python3 -m gns.train --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" -ntraining_steps=100
# For mesh-based domain,
python3 -m meshnet.train --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" -ntraining_steps=100
```

> Resume training

To resume training specify `model_file` and `train_state_file`:

```shell
# For particulate domain,
python3 -m gns.train --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>"  --model_file="model.pt" --train_state_file="train_state.pt" -ntraining_steps=100
# For mesh-based domain,
python3 -m meshnet.train --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>"  --model_file="model.pt" --train_state_file="train_state.pt" -ntraining_steps=100
```

> Rollout prediction
```shell
# For particulate domain,
python3 -m gns.train --mode="rollout" --data_path="<input-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" --model_file="model.pt" --train_state_file="train_state.pt"
# For mesh-based domain,
python3 -m meshnet.train --mode="rollout" --data_path="<input-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" --model_file="model.pt" --train_state_file="train_state.pt"
```

> Render
```shell
# For particulate domain,
python3 -m gns.render_rollout --output_mode="gif" --rollout_dir="<path-containing-rollout-file>" --rollout_name="<name-of-rollout-file>"
# For mesh-based domain,
python3 -m gns.render --rollout_dir="<path-containing-rollout-file>" --rollout_name="<name-of-rollout-file>"
```

In particulate domain, the renderer also writes `.vtu` files to visualize in ParaView.

![Sand rollout](docs/img/rollout_0.gif)
> GNS prediction of Sand rollout after training for 2 million steps.

In mesh-based domain, the renderer writes `.gif` animation.

![Fluid flow rollout](docs/img/meshnet.gif)
> Meshnet GNS prediction of cylinder flow after training for 1 million steps.


## Command line arguments details
<details>
<summary>`train.py` in GNS (particulate domain) </summary>

**mode (Enum)** 

This flag is used to set the operation mode for the script. It can take one of three values; 'train', 'valid', or 'rollout'.

**batch_size (Integer)**

Batch size for training.

**noise_std (Float)** 

Standard deviation of the noise when training.

**data_path (String)** 

Specifies the directory path where the dataset is located. 
The dataset is expected to be in a specific format (e.g., .npz files).
It should contain `metadata.json`.
If `--mode` is training, the directory should contain `train.npz`.
If `--mode` is testing (rollout), the directory should contain `test.npz`.
If `--mode` is valid, the directory should contain `valid.npz`.

**model_path (String)** 

The directory path where the trained model checkpoints are saved during training or loaded from during validation/rollout.

**output_path (String)** 

Defines the directory where the outputs (e.g., rollouts) are saved, 
when the `--mode` is set to rollout.
This is particularly relevant in the rollout mode where the predictions of the model are stored.

**output_filename (String)** 

Base filename to use when saving outputs during rollout.
Default is "rollout", and the output will be saved as `rollout.pkl` in `output_path`. 
It is not intended to include the file extension.

**model_file (String)** 

The filename of the model checkpoint to load for validation or rollout (e.g., model-10000.pt). 
It supports a special value "latest" to automatically select the newest checkpoint file. 
This flexibility facilitates the evaluation of models at different stages of training.

**train_state_file (String)** 

Similar to model_file, but for loading the training state (e.g., optimizer state).
It supports a special value "latest" to automatically select the newest checkpoint file. 
(e.g., training_state-10000.pt)

**ntraining_steps (Integer)** 

The total number of training steps to execute before stopping.

**nsave_steps (Integer)** 

Interval at which the model and training state are saved.

**lr_init (Float)** 

Initial learning rate.

**lr_decay (Float)** 

How much the learning rate should decay over time.

**lr_decay_steps (Integer)** 

Steps at which learning rate should decay.

**cuda_device_number (Integer)** 

Base CUDA device (zero indexed).
Default is None so default CUDA device will be used.

**n_gpus (Integer)** 

Number of GPUs to use for training.
</details>



<details>
<summary>`train.py` in MeshNet (mesh-based domain) </summary>

**mode (String)**

This flag is used to set the operation mode for the script. It can take one of three values; 'train', 'valid', or 'rollout'.

**batch_size (Integer)** 

Batch size for training.

**data_path (String)**

Specifies the directory path where the dataset is located. 
The dataset is expected to be in a specific format (e.g., .npz files).
If `--mode` is training, the directory should contain `train.npz`.
If `--mode` is testing (rollout), the directory should contain `test.npz`.
If `--mode` is valid, the directory should contain `valid.npz`.

**model_path (String)** 

The directory path where the trained model checkpoints are saved during training or loaded from during validation/rollout.

**output_path (String)**

Defines the directory where the outputs (e.g., rollouts) are saved, 
when the `--mode` is set to rollout.
This is particularly relevant in the rollout mode where the predictions of the model are stored.

**model_file (String)**

The filename of the model checkpoint to load for validation or rollout (e.g., model-10000.pt). 
It supports a special value "latest" to automatically select the newest checkpoint file. 
This flexibility facilitates the evaluation of models at different stages of training.

**train_state_file (String)**

Similar to model_file, but for loading the training state (e.g., optimizer state).
It supports a special value "latest" to automatically select the newest checkpoint file. 
(e.g., training_state-10000.pt)

**cuda_device_number (Integer)**

Allows specifying a particular CUDA device for training or evaluation, enabling the use of specific GPUs in multi-GPU setups.

**rollout_filename (String)**

Base name for saving rollout files. The actual filenames will append an index to this base name.

**ntraining_steps (Integer)**

The total number of training steps to execute before stopping.

**nsave_steps (Integer)**

Interval at which the model and training state are saved.

</details>

## Datasets
### Particulate domain:
We use the numpy `.npz` format for storing positional data for GNS training.  The `.npz` format includes a list of tuples of arbitrary length where each tuple corresponds to a differenet training trajectory and is of the form `(position, particle_type)`.  The data loader provides `INPUT_SEQUENCE_LENGTH` positions, set equal to six by default, to provide the GNS with the last `INPUT_SEQUENCE_LENGTH` minus one positions as input to predict the position at the next time step.  The `position` is a 3-D tensor of shape `(n_time_steps, n_particles, n_dimensions)` and `particle_type` is a 1-D tensor of shape `(n_particles)`.  

The dataset contains:

* Metadata file with dataset information `(sequence length, dimensionality, box bounds, default connectivity radius, statistics for normalization, ...)`:

```
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
* npz containing data for all trajectories `(particle types, positions, global context, ...)`:

Training datasets for Sand, SandRamps, and WaterDropSample are available on [DesignSafe Data Depot](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3702) [@vantassel2022gnsdata].

We provide the following datasets:
  * `WaterDropSample` (smallest dataset)
  * `Sand`
  * `SandRamps`

Download the dataset [DesignSafe DataDepot](https://doi.org/10.17603/ds2-0phb-dg64). If you are using this dataset please cite [Vantassel and Kumar., 2022](https://github.com/geoelements/gns#dataset)

### Mesh-based domain:
We also use the numpy `.npz` format for storing data for training meshnet GNS.

The dataset contains:
* npz containing python dictionary describing mesh data and relevant dynamics at mesh nodes for all trajectories. The dictionary includes `{pos: (ntimestep, nnodes, ndims), node_type: (ntimestep, nnodes, ntypes), velocity: (ntimestep, nnodes, ndims), pressure: (ntimestep, nnodes, 1), cells: (ntimestep, ncells, 3)}`

The dataset is shared on [DesignSafe DataDepot](https://doi.org/10.17603/ds2-fzg7-1719). If you are using this dataset please cite [Kumar and Choi., 2023](https://github.com/geoelements/gns#dataset)

## Installation

GNS uses [pytorch geometric](https://www.pyg.org/) and [CUDA](https://developer.nvidia.com/cuda-downloads). These packages have specific requirements, please see [PyG installation]((https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for details. 

> CPU-only installation on Linux

```shell
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y pyg -c pyg
conda install -y pytorch-cluster -c pyg
conda install -y absl-py -c anaconda 
conda install -y numpy dm-tree matplotlib-base pyevtk -c conda-forge 
```
You can use the [WaterDropletSample](https://github.com/geoelements/gns-sample) dataset to check if your `gns` code is working correctly.

To test the code you can run:

```
pytest test/
```

To test on the small waterdroplet sample:

```
git clone https://github.com/geoelements/gns-sample

TMP_DIR="./gns-sample"
DATASET_NAME="WaterDropSample"

mkdir -p ${TMP_DIR}/${DATASET_NAME}/models/
mkdir -p ${TMP_DIR}/${DATASET_NAME}/rollout/

DATA_PATH="${TMP_DIR}/${DATASET_NAME}/dataset/"
MODEL_PATH="${TMP_DIR}/${DATASET_NAME}/models/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET_NAME}/rollout/"

python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=10
```


### Building GNS environment on TACC (LS6 and Frontera)

- to setup a virtualenv

```shell
sh ./build_venv.sh
```

- check tests run sucessfully.
- start your environment

```shell
source start_venv.sh 
```

### Inspiration
PyTorch version of Graph Network Simulator and Mesh Graph Network Simulator are based on:
* [https://arxiv.org/abs/2002.09405](https://arxiv.org/abs/2002.09405) and [https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate)
* [https://arxiv.org/abs/2010.03409](https://arxiv.org/abs/2002.09405) and [https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets)
* [https://github.com/echowve/meshGraphNets_pytorch](https://github.com/echowve/meshGraphNets_pytorch)

### Acknowledgement
This code is based upon work supported by the National Science Foundation under Grant OAC-2103937.

### Citation

#### Repo
Kumar, K., & Vantassel, J. (2023). GNS: A generalizable Graph Neural Network-based simulator for particulate and fluid modeling. Journal of Open Source Software, 8(88), 5025. https://doi.org/10.21105/joss.05025

#### Dataset
* Vantassel, Joseph; Kumar, Krishna (2022) “Graph Network Simulator Datasets.” DesignSafe-CI. https://doi.org/10.17603/ds2-0phb-dg64 v1 
* Kumar, K., Y. Choi. (2023) "Cylinder flow with graph neural network-based simulator." DesignSafe-CI. https://doi.org/10.17603/ds2-fzg7-1719

