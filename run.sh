#!/bin/bash

# Fail on any error.
set -e

# Display commands being run.
set -x

TMP_DIR="../"

virtualenv --python=python3.6 "${TMP_DIR}/gns/venv"
source "${TMP_DIR}/gns/venv/bin/activate"

# Install dependencies.
pip install --upgrade -r requirements.txt


# Run some training and evaluation in one of the dataset samples.
# Download a sample of a dataset.
DATASET_NAME="WaterDropSample"

# bash ./download_dataset.sh ${DATASET_NAME} "${TMP_DIR}/datasets"

# Train for a few steps.
DATA_PATH="${TMP_DIR}/datasets/${DATASET_NAME}/"
MODEL_PATH="${TMP_DIR}/models/${DATASET_NAME}/"
python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=10 --mode='train'

# Evaluate on validation split.
python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="model.pt" --mode="valid"

# Generate test rollouts.
ROLLOUT_PATH="${TMP_DIR}/rollouts/${DATASET_NAME}/"
mkdir -p ${ROLLOUT_PATH}
python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="model.pt" --output_path=${ROLLOUT_PATH} --mode='rollout'

# Plot the first rollout.
python -m gns.render_rollout --rollout_path="${ROLLOUT_PATH}/rollout_0.pkl" --block_on_show=False

# Clean up.
rm -r ${TMP_DIR}
