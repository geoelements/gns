import collections
import json
import numpy as np
import os
import torch
import pickle
import glob
import re

import tree

from absl import flags
from absl import app


from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')

flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")

# yc:
flags.DEFINE_string('loss_mode', 'accel', help="Options for loss functions: 'acceleration' or 'position'")
flags.DEFINE_float('alpha', 0.5, help='Weight value for positional loss')

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.tensor,
        particle_types: torch.tensor,
        n_particles_per_example: torch.tensor,
        nsteps: int,
        device):
  """Rolls out a trajectory by applying the model in sequence.

  Args:
    simulator: Learned simulator.
    features: Torch tensor features.
    nsteps: Number of steps.
  """
  initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

  current_positions = initial_positions
  predictions = []

  for step in range(nsteps):
    # Get next position with shape (nnodes, dim)
    next_position = simulator.predict_positions(
        current_positions,
        nparticles_per_example=[n_particles_per_example],
        particle_types=particle_types,
    )

    # Update kinematic particles from prescribed trajectory.
    kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone().detach().to(device)
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
      'particle_types': particle_types.cpu().numpy(),
  }

  return output_dict, loss


def predict(
        simulator: learned_simulator.LearnedSimulator,
        metadata: json,
        device: str):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.
    metadata: Metadata for test set.

  """

  # Load simulator
  if os.path.exists(FLAGS.model_path + FLAGS.model_file):
    simulator.load(FLAGS.model_path + FLAGS.model_file)
  else:
    train(simulator)
  
  simulator.to(device)
  simulator.eval()

  # Output path
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  # Use `valid`` set for eval mode if not use `test`
  split = 'test' if FLAGS.mode == 'rollout' else 'valid'

  ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")

  eval_loss = []
  with torch.no_grad():
    for example_i, (positions, particle_type, n_particles_per_example) in enumerate(ds):
      positions.to(device)
      particle_type.to(device)
      n_particles_per_example = torch.tensor([int(n_particles_per_example)], dtype=torch.int32).to(device)

      nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
      # Predict example rollout
      example_rollout, loss = rollout(simulator, positions.to(device), particle_type.to(device),
                                      n_particles_per_example.to(device), nsteps, device)

      example_rollout['metadata'] = metadata
      print("Predicting example {} loss: {}".format(example_i, loss.mean()))
      eval_loss.append(torch.flatten(loss))
      
      # Save rollout in testing
      if FLAGS.mode == 'rollout':
        example_rollout['metadata'] = metadata
        filename = f'rollout_{example_i}.pkl'
        filename = os.path.join(FLAGS.output_path, filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)

  print("Mean loss on rollout prediction: {}".format(
      torch.mean(torch.cat(eval_loss))))

def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)

def train(
        simulator: learned_simulator.LearnedSimulator,
        device: str):
  """Train the model.

  Args:
    simulator: Get LearnedSimulator.
    loss_function: choose how to evaluate loss. One is acceleration based loss and the other is
        position based weighed loss.
  """
  optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init)
  step = 0
  # If model_path does not exist create new directory and begin training.
  model_path = FLAGS.model_path
  if not os.path.exists(model_path):
    os.makedirs(model_path)
    

  # If model_path does exist and model_file and train_state_file exist continue training.
  if FLAGS.model_file is not None:

    if FLAGS.model_file == "latest" and FLAGS.train_state_file == "latest":
      # find the latest model, assumes model and train_state files are in step.
      fnames = glob.glob(f"{model_path}*model*pt")
      max_model_number = 0
      expr = re.compile(".*model-(\d+).pt")
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      # reset names to point to the latest.
      FLAGS.model_file = f"model-{max_model_number}.pt"
      FLAGS.train_state_file = f"train_state-{max_model_number}.pt"

    if os.path.exists(model_path + FLAGS.model_file) and os.path.exists(model_path + FLAGS.train_state_file):
      # load model
      simulator.load(model_path + FLAGS.model_file)

      # load train state
      train_state = torch.load(model_path + FLAGS.train_state_file)
      # set optimizer state
      optimizer = torch.optim.Adam(simulator.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, device)
      # set global train state
      step = train_state["global_train_state"].pop("step")
 
    else:
      msg = f"Specified model_file {model_path + FLAGS.model_file} and train_state_file {model_path + FLAGS.train_state_file} not found."
      raise FileNotFoundError(msg) 

  simulator.train()
  simulator.to(device)

  ds = data_loader.get_data_loader_by_samples(path=f"{FLAGS.data_path}train.npz",
                                              input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                              batch_size=FLAGS.batch_size,
                                            )

  print(f"device = {device}")
  not_reached_nsteps = True
  try:
    while not_reached_nsteps:  
        position.to(device)
        particle_type.to(device)
        n_particles_per_example.to(device)
        labels.to(device)

        # TODO (jpv): Move noise addition to data_loader
        # Sample the noise to add to the inputs to the model during training.
        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position, noise_std_last_step=FLAGS.noise_std).to(device)
        non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device)
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

        # Get the predictions and target accelerations.
        pred_acc, target_acc = simulator.predict_accelerations(
            next_positions=labels.to(device),
            position_sequence_noise=sampled_noise.to(device),
            position_sequence=position.to(device),
            nparticles_per_example=n_particles_per_example.to(device),
            particle_types=particle_type.to(device))

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

        print('Training step: {}/{}. Loss: {}.'.format(step,
                                                       FLAGS.ntraining_steps,
                                                     loss))
        # Save model state
        if step % FLAGS.nsave_steps == 0:
          simulator.save(model_path + 'model-'+str(step)+'.pt')
          train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
          torch.save(train_state, f"{model_path}train_state-{step}.pt")

        # Complete training
        if (step >= FLAGS.ntraining_steps):
          not_reached_nsteps = False
          break

        step += 1

  except KeyboardInterrupt:
    pass

  simulator.save(model_path + 'model-'+str(step)+'.pt')
  train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
  torch.save(train_state, f"{model_path}train_state-{step}.pt")


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
    device: PyTorch device 'cpu' or 'cuda'.
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
  """Train or evaluates the model.

  """
  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
    device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')

  # Read metadata
  metadata = reading_utils.read_metadata(FLAGS.data_path)
  simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device)
  if FLAGS.mode == 'train':
    train(simulator, device)
  elif FLAGS.mode in ['valid', 'rollout']:
    predict(simulator, metadata, device)


if __name__ == '__main__':
  app.run(main)
