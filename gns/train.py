import collections
import functools
import os
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import reading_utils
import tree

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_steps', int(
    2e7), help='Number of steps of training.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')


FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3


def prepare_inputs(tensor_dict):
  """Prepares a single stack of inputs by calculating inputs and targets.

  Computes n_particles_per_example, which is a tensor that contains information
  about how to partition the axis - i.e. which nodes belong to which graph.

  Adds a batch axis to `n_particles_per_example` and `step_context` so they can
  later be batched using `batch_concat`. This batch will be the same as if the
  elements had been batched via stacking.

  Note that all other tensors have a variable size particle axis,
  and in this case they will simply be concatenated along that
  axis.

  Args:
    tensor_dict: A dict of tensors containing positions, and step context (
    if available).

  Returns:
    A tuple of input features and target positions.

  """
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])

  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]

  # Remove the target from the input.
  tensor_dict['position'] = pos[:, :-1]

  # Compute the number of particles per example.
  num_particles = tf.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

  if 'step_context' in tensor_dict:
    # Take the input global context. We have a stack of global contexts,
    # and we take the penultimate since the final is the target.
    tensor_dict['step_context'] = tensor_dict['step_context'][-2]
    # Add an extra dimension for stacking via concat.
    tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
  return tensor_dict, target_position


def prepare_rollout_inputs(context, features):
  """Prepares an inputs trajectory for rollout."""
  out_dict = {**context}
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tf.transpose(features['position'], [1, 0, 2])
  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]
  # Remove the target from the input.
  out_dict['position'] = pos[:, :-1]
  # Compute the number of nodes
  out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
  if 'step_context' in features:
    out_dict['step_context'] = features['step_context']
  out_dict['is_trajectory'] = tf.constant([True], tf.bool)
  return out_dict, target_position


def batch_concat(dataset, batch_size):
  """We implement batching as concatenating on the leading axis."""

  # We create a dataset of datasets of length batch_size.
  windowed_ds = dataset.window(batch_size)

  # The plan is then to reduce every nested dataset by concatenating. We can
  # do this using tf.data.Dataset.reduce. This requires an initial state, and
  # then incrementally reduces by running through the dataset

  # Get initial state. In this case this will be empty tensors of the
  # correct shape.
  initial_state = tree.map_structure(
      lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
          shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
      dataset.element_spec)

  # We run through the nest and concatenate each entry with the previous state.
  def reduce_window(initial_state, ds):
    return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

  return windowed_ds.map(
      lambda *x: tree.map_structure(reduce_window, initial_state, x))


def prepare_input_data(
        data_path: str,
        batch_size: int = 2,
        mode: str = 'train',
        split: str = 'train'):
  """Prepares the input data for learning simulation from tfrecord.

  Args:
    data_path: the path to the dataset directory.
    batch_size: the number of graphs in a batch.
    mode: either 'train' or 'rollout'
    split: either 'train', 'valid' or 'test.

  Returns:
    The input data for the learning simulation model.
  """
  # Loads the metadata of the dataset.
  metadata = reading_utils.read_metadata(data_path)
  # Create a tf.data.Dataset from the TFRecord.
  ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
  ds = ds.map(functools.partial(
      reading_utils.parse_serialized_simulation_example, metadata=metadata))

  # Splits an entire trajectory into chunks of 7 steps.
  # Previous 5 velocities, current velocity and target.
  split_with_window = functools.partial(
      reading_utils.split_trajectory,
      window_length=INPUT_SEQUENCE_LENGTH + 1)
  ds = ds.flat_map(split_with_window)
  # Splits a chunk into input steps and target steps
  ds = ds.map(prepare_inputs)
  # If in train mode, repeat dataset forever and shuffle.
  if mode == 'train':
    ds = ds.repeat()
    ds = ds.shuffle(512)
  # Custom batching on the leading axis.
  ds = batch_concat(ds, batch_size)
  ds = tfds.as_numpy(ds)

  return ds
