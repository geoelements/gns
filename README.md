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
