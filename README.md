# Graph Network Simulator

## Run GNS
> Training
```shell
python3 -m gns.train --data_path='../datasets/WaterDropSample/'
```

> Rollout
```shell
python3 -m gns.train --mode='rollout' --data_path='../datasets/WaterDropSample/' --model_path='../models/WaterDropSample/' --output_path='../rollouts'
```

> Render
```shell
 python3 -m gns.render_rollout --rollout_path='../rollouts/WaterDropSample/rollout_0.pkl' 
```

### Configure environment on TACC Frontera

- setup a virtualenv
```
ml use /work2/02604/ajs2987/frontera/apps/modulefiles
ml cuda/11.1
module load python3/3.9.2

python3 -m virtualenv venv
source venv/bin/activate

which pip3
pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip3 install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu111.html --no-binary torch-spline-conv
pip3 install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
```

- test pytorch with gpu

```python
import torch
print(torch.cuda.is_available()) # --> True
=======
## Run GNS
> Training

```shell
python3 -m gns.train --data_path='../datasets/WaterDropSample/'
```
## Datasets

Datasets are available to download via:

* Metadata file with dataset information (sequence length, dimensionality, box bounds, default connectivity radius, statistics for normalization, ...):

  `https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{DATASET_NAME}/metadata.json`

* TFRecords containing data for all trajectories (particle types, positions, global context, ...):

  `https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{DATASET_NAME}/{DATASET_SPLIT}.tfrecord`

Where:

* `{DATASET_SPLIT}` is one of:
  * `train`
  * `valid`
  * `test`

* `{DATASET_NAME}` one of the datasets following the naming used in the paper:
  * `WaterDrop`
  * `Water`
  * `Sand`
  * `Goop`
  * `MultiMaterial`
  * `RandomFloor`
  * `WaterRamps`
  * `SandRamps`
  * `FluidShake`
  * `FluidShakeBox`
  * `Continuous`
  * `WaterDrop-XL`
  * `Water-3D`
  * `Sand-3D`
  * `Goop-3D`

The provided script `./download_dataset.sh` may be used to download all files from each dataset into a folder given its name.

An additional smaller dataset `WaterDropSample`, which includes only the first two trajectories of `WaterDrop` for each split, is provided for debugging purposes.

### Download dataset (e.g., WaterRamps)


```shell
    mkdir -p /tmp/datasets
    bash ./download_dataset.sh WaterRamps /tmp/datasets
```
