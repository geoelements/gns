#!/bin/bash
# Copyright 2020 Deepmind Technologies Limited.
# Copyright 2021 Geoelements.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error.
set -e

# Display commands being run.
set -x

TMP_DIR="../"

virtualenv --python=python3.6 "${TMP_DIR}/gns/venv"
source "${TMP_DIR}/gns/venv/bin/activate"

# Install dependencies.
pip install --upgrade -r requirements.txt

# Run the simple demo with dummy inputs.
#python -m gns.model_demo

# Run some training and evaluation in one of the dataset samples.

# Download a sample of a dataset.
DATASET_NAME="WaterDropSample"

bash ./download_dataset.sh ${DATASET_NAME} "${TMP_DIR}/datasets"

# Train for a few steps.
DATA_PATH="${TMP_DIR}/datasets/${DATASET_NAME}"
MODEL_PATH="${TMP_DIR}/models/${DATASET_NAME}"
python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=10 --mode='train'

# Evaluate on validation split.
python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --mode="valid"

# Generate test rollouts.
ROLLOUT_PATH="${TMP_DIR}/rollouts/${DATASET_NAME}"
mkdir -p ${ROLLOUT_PATH}
python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --output_path=${ROLLOUT_PATH} --mode='rollout'

# Plot the first rollout.
python -m gns.render_rollout --rollout_path="${ROLLOUT_PATH}/rollout_test_0.pkl" --block_on_show=False

# Clean up.
rm -r ${TMP_DIR}
