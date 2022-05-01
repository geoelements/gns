# Graph Network Simulator
> PyTorch version of Graph Network Simulator based on [https://arxiv.org/abs/2002.09405](https://arxiv.org/abs/2002.09405) and [https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate).

## Run GNS
> Training
```shell
export DATASET_NAME="Sand"
export WORK_DIR=${WORK_DIR}/
python3 -m gns.train --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}/" -ntraining_steps=100
```

> Resume training

To resume training specify `model_file` and `train_state_file`:

```shell
export DATASET_NAME="Sand"
export WORK_DIR=${WORK_DIR}/
python3 -m gns.train --data_path="${WORK_DIR}/datasets/${DATASET_NAME}/" --model_path="${WORK_DIR}/models/${DATASET_NAME}/" --output_path="${WORK_DIR}/rollouts/${DATASET_NAME}/" --model_file="model.pt" --train_state_file="train_state.pt" -ntraining_steps=100
```

> Rollout
```shell
python3 -m gns.train --mode='rollout' --data_path='${WORK_DIR}/datasets/${DATASET_NAME}/' --model_path='${WORK_DIR}/models/${DATASET_NAME}/' --model_file='model.pt' --output_path='${WORK_DIR}/rollouts'
```

> Render
```shell
 python3 -m gns.render_rollout --rollout_path='${WORK_DIR}/rollouts/${DATASET_NAME}/rollout_0.pkl' 
```

![Sand rollout](figs/rollout_0.gif)
> GNS prediction of Sand rollout after training for 2 million steps.

## Datasets

The data loader provided with this PyTorch implementation utilizes the more general `.npz` format. The `.npz` format includes a list of
tuples of arbitrary length where each tuple is for a different training trajectory
and is of the form `(position, particle_type)`. `position` is a 3-D tensor of
shape `(n_time_steps, n_particles, n_dimensions)` and `particle_type` is
a 1-D tensor of shape `(n_particles)`.  

The dataset contains:

* Metadata file with dataset information (sequence length, dimensionality, box bounds, default connectivity radius, statistics for normalization, ...):

* npz containing data for all trajectories (particle types, positions, global context, ...):

We provide the following datasets:
  * `WaterDropSample` (smallest dataset)
  * `Sand`
  * `SandRamps`

Download the dataset from [UT Box](https://utexas.app.box.com/s/awryzbj5oexa18f5njcnw7zr7uf4w80q)


## Building environment on TACC LS6 and Frontera

- to setup a virtualenv

```shell
sh ./build_venv.sh
```

- check tests run sucessfully.
- start your environment

```shell
source start_venv.sh 
```

