import collections
import functools
import json
import numpy as np
import os
import torch

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tree

from absl import flags
from absl import logging
from absl import app


from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')

flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', 'models/',
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')

flags.DEFINE_string('device', 'cpu', help='Device to train `cpu` or `cuda`.')

flags.DEFINE_integer('ntraining_steps', int(2e7),
                     help='Number of training steps.')
flags.DEFINE_integer('neval_steps', int(20), help='Number of eval steps.')
flags.DEFINE_integer('nsave_steps', int(100), help='Frequency to save states.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(
    5e6), help='Learning rate decay steps.')

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
  nparticles = tf.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = nparticles[tf.newaxis]

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
  # Position is encoded as [sequence_length, nparticles, dim] but the model
  # expects [nparticles, sequence_length, dim].
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


def rollout(
        simulator: learned_simulator.LearnedSimulator,
        features: torch.tensor,
        nsteps: int,
        device: str):
  """Rolls out a trajectory by applying the model in sequence.

  Args:
    simulator: Learned simulator.
    features: Torch tensor features.
    nsteps: Number of steps.
    device: cpu or cuda device.
  """
  initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]

  current_positions = initial_positions
  predictions = []

  for step in range(nsteps):
    # Get next position with shape (nnodes, dim)
    next_position = simulator.predict_positions(
        current_positions,
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_type'],
    )

    # Update kinematic particles from prescribed trajectory.
    kinematic_mask = (features['particle_type'] ==
                      3).clone().detach().to(device)
    next_position_ground_truth = ground_truth_positions[:, step]
    kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, 2)
    next_position = torch.where(
        kinematic_mask, next_position_ground_truth, next_position)
    predictions.append(next_position)

    # Shift `current_positions`, removing the oldest position in the sequence
    # and appending the next position at the end.
    current_positions = torch.cat(
        [current_positions[:, 1:], next_position[:, None, :]], dim=1)

  # Predictions with shape (time, nnodes, dim)
  predictions = torch.stack(predictions)
  ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

  loss = (predictions - ground_truth_positions) ** 2

  output_dict = {
      'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
      'predicted_rollout': predictions.cpu().numpy(),
      'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
      'particle_types': features['particle_type'].cpu().numpy(),
  }

  return output_dict, loss


def train(simulator):

  # Model path
  model_path = FLAGS.model_path
  if not os.path.exists(FLAGS.model_path):
    os.mkdir(FLAGS.model_path)

  device = FLAGS.device

  # Learning rate parameters
  lr_new = FLAGS.lr_init
  optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init)

  ds = prepare_input_data(FLAGS.data_path,
                          batch_size=FLAGS.batch_size)

  step = 0
  try:
    for features, labels in ds:
      features['position'] = torch.tensor(
          features['position']).to(device)
      features['n_particles_per_example'] = torch.tensor(
          features['n_particles_per_example']).to(device)
      features['particle_type'] = torch.tensor(
          features['particle_type']).to(device)
      labels = torch.tensor(labels).to(device)

      # Sample the noise to add to the inputs to the model during training.
      sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
          features['position'], noise_std_last_step=FLAGS.noise_std).to(device)
      non_kinematic_mask = (
          features['particle_type'] != 3).clone().detach().to(device)
      sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

      # Get the predictions and target accelerations.
      pred_acc, target_acc = simulator.predict_accelerations(
          next_positions=labels,
          position_sequence_noise=sampled_noise,
          position_sequence=features['position'],
          nparticles_per_example=features['n_particles_per_example'],
          particle_types=features['particle_type'])

      # Calculate the loss and mask out loss on kinematic particles
      loss = (pred_acc - target_acc) ** 2
      loss = loss.sum(dim=-1)
      num_non_kinematic = non_kinematic_mask.sum()
      loss = torch.where(non_kinematic_mask.bool(),
                         loss, torch.zeros_like(loss))
      loss = loss.sum() / num_non_kinematic

      # Computes the gradient of loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Update learning rate
      lr_new = FLAGS.lr_init * (FLAGS.lr_decay ** (step/FLAGS.lr_decay_steps))
      for param in optimizer.param_groups:
        param['lr'] = lr_new

      step += 1
      print('Training step: {}/{}. Loss: {}.'.format(step,
                                                     FLAGS.ntraining_steps,
                                                     loss))
      # Save model state
      if step % FLAGS.nsave_steps == 0:
        simulator.save(model_path + 'model.pt')

      # Complete training
      if (step > FLAGS.ntraining_steps):
        break

  except KeyboardInterrupt:
    pass

  simulator.save(model_path + 'finalmodel.pt')


def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: str) -> learned_simulator.LearnedSimulator:
  """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    acc_noise_std: Acceleration noise std deviation.
    vel_noise_std: Velocity noise std deviation.
    device: 'cpu' or 'cuda'.
  """

  # Normalization stats
  normalization_stats = {
      'acceleration': {
          'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 +
                            acc_noise_std**2).to(device),
      },
      'velocity': {
          'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 +
                            vel_noise_std**2).to(device),
      },
  }

  simulator = learned_simulator.LearnedSimulator(
      particle_dimensions=metadata['dim'],
      nnode_in=30,
      nedge_in=3,
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      device=device)

  return simulator


def main(_):
  """Train or evaluates the model."""
  # Read metadata
  metadata = reading_utils.read_metadata(FLAGS.data_path)
  simulator = _get_simulator(metadata, FLAGS.noise_std,
                             FLAGS.noise_std, FLAGS.device)
  train(simulator)


if __name__ == '__main__':
  app.run(main)
