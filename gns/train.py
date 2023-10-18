import collections
import json
import os
import pickle
import glob
import re
import sys

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from absl import flags
from absl import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('output_filename', 'rollout', help='Base name for saving the rollout')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')

flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.tensor,
        particle_types: torch.tensor,
        material_property: torch.tensor,
        n_particles_per_example: torch.tensor,
        nsteps: int,
        device: torch.device):
  """
  Rolls out a trajectory by applying the model in sequence.

  Args:
    simulator: Learned simulator.
    position: Positions of particles (timesteps, nparticles, ndims)
    particle_types: Particles types with shape (nparticles)
    material_property: Friction angle normalized by tan() with shape (nparticles)
    n_particles_per_example
    nsteps: Number of steps.
    device: torch device.
  """

  initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

  current_positions = initial_positions
  predictions = []

  for step in tqdm(range(nsteps), total=nsteps):
    # Get next position with shape (nnodes, dim)
    next_position = simulator.predict_positions(
        current_positions,
        nparticles_per_example=[n_particles_per_example],
        particle_types=particle_types,
        material_property=material_property
    )

    # Update kinematic particles from prescribed trajectory.
    kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone().detach().to(device)
    next_position_ground_truth = ground_truth_positions[:, step]
    kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
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
      'material_property': material_property.cpu().numpy() if material_property is not None else None
  }

  return output_dict, loss


def predict(device: str):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
  # Read metadata
  metadata = reading_utils.read_metadata(FLAGS.data_path, "rollout")
  simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device)

  # Load simulator
  if os.path.exists(FLAGS.model_path + FLAGS.model_file):
    simulator.load(FLAGS.model_path + FLAGS.model_file)
  else:
    raise Exception(f"Model does not exist at {FLAGS.model_path + FLAGS.model_file}")

  simulator.to(device)
  simulator.eval()

  # Output path
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  # Use `valid`` set for eval mode if not use `test`
  split = 'test' if FLAGS.mode == 'rollout' else 'valid'

  # Get dataset
  ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")
  # See if our dataset has material property as feature
  if len(ds.dataset._data[0]) == 3:  # `ds` has (positions, particle_type, material_property)
    material_property_as_feature = True
  elif len(ds.dataset._data[0]) == 2:  # `ds` only has (positions, particle_type)
    material_property_as_feature = False
  else:
    raise NotImplementedError

  eval_loss = []
  with torch.no_grad():
    for example_i, features in enumerate(ds):
      positions = features[0].to(device)
      if metadata['sequence_length'] is not None:
        # If `sequence_length` is predefined in metadata,
        nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
      else:
        # If no predefined `sequence_length`, then get the sequence length
        sequence_length = positions.shape[1]
        nsteps = sequence_length - INPUT_SEQUENCE_LENGTH
      particle_type = features[1].to(device)
      if material_property_as_feature:
        material_property = features[2].to(device)
        n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device)
      else:
        material_property = None
        n_particles_per_example = torch.tensor([int(features[2])], dtype=torch.int32).to(device)

      # Predict example rollout
      example_rollout, loss = rollout(simulator,
                                      positions,
                                      particle_type,
                                      material_property,
                                      n_particles_per_example,
                                      nsteps,
                                      device)

      example_rollout['metadata'] = metadata
      print("Predicting example {} loss: {}".format(example_i, loss.mean()))
      eval_loss.append(torch.flatten(loss))

      # Save rollout in testing
      if FLAGS.mode == 'rollout':
        example_rollout['metadata'] = metadata
        example_rollout['loss'] = loss.mean()
        filename = f'{FLAGS.output_filename}_ex{example_i}.pkl'
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

def train(rank, flags, world_size, device):
  """Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
    device: torch device type
  """
  if device == torch.device("cuda"):
    distribute.setup(rank, world_size, device)
    device_id = rank
  else:
    device_id = device

  # Read metadata
  metadata = reading_utils.read_metadata(flags["data_path"], "train")

  # Get simulator and optimizer
  if device == torch.device("cuda"):
    serial_simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], rank)
    simulator = DDP(serial_simulator.to(rank), device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"]*world_size)
  else:
    simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"] * world_size)
  step = 0

  # If model_path does exist and model_file and train_state_file exist continue training.
  if flags["model_file"] is not None:

    if flags["model_file"] == "latest" and flags["train_state_file"] == "latest":
      # find the latest model, assumes model and train_state files are in step.
      fnames = glob.glob(f'{flags["model_path"]}*model*pt')
      max_model_number = 0
      expr = re.compile(".*model-(\d+).pt")
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      # reset names to point to the latest.
      flags["model_file"] = f"model-{max_model_number}.pt"
      flags["train_state_file"] = f"train_state-{max_model_number}.pt"

    if os.path.exists(flags["model_path"] + flags["model_file"]) and os.path.exists(flags["model_path"] + flags["train_state_file"]):
      # load model
      if device == torch.device("cuda"):
        simulator.module.load(flags["model_path"] + flags["model_file"])
      else:
        simulator.load(flags["model_path"] + flags["model_file"])

      # load train state
      train_state = torch.load(flags["model_path"] + flags["train_state_file"])
      # set optimizer state
      optimizer = torch.optim.Adam(
        simulator.module.parameters() if device == torch.device("cuda") else simulator.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, device_id)
      # set global train state
      step = train_state["global_train_state"].pop("step")

    else:
      msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
      raise FileNotFoundError(msg)

  simulator.train()
  simulator.to(device_id)

  if device == torch.device("cuda"):
    dl = distribute.get_data_distributed_dataloader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                               input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                               batch_size=flags["batch_size"])
  else:
    dl = data_loader.get_data_loader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                batch_size=flags["batch_size"])
  n_features = len(dl.dataset._data[0])

  print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")
  not_reached_nsteps = True
  try:
    while not_reached_nsteps:
      if device == torch.device("cuda"):
        torch.distributed.barrier()
      else:
        pass
      for example in dl:  # ((position, particle_type, material_property, n_particles_per_example), labels) are in dl
        position = example[0][0].to(device_id)
        particle_type = example[0][1].to(device_id)
        if n_features == 3:  # if dl includes material_property
          material_property = example[0][2].to(device_id)
          n_particles_per_example = example[0][3].to(device_id)
        elif n_features == 2:
          n_particles_per_example = example[0][2].to(device_id)
        else:
          raise NotImplementedError
        labels = example[1].to(device_id)

        n_particles_per_example.to(device_id)
        labels.to(device_id)

        # TODO (jpv): Move noise addition to data_loader
        # Sample the noise to add to the inputs to the model during training.
        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position, noise_std_last_step=flags["noise_std"]).to(device_id)
        non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device_id)
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

        # Get the predictions and target accelerations.
        if device == torch.device("cuda"):
          pred_acc, target_acc = simulator.module.predict_accelerations(
            next_positions=labels.to(rank),
            position_sequence_noise=sampled_noise.to(rank),
            position_sequence=position.to(rank),
            nparticles_per_example=n_particles_per_example.to(rank),
            particle_types=particle_type.to(rank),
            material_property=material_property.to(rank) if n_features == 3 else None
          )
        else:
          pred_acc, target_acc = simulator.predict_accelerations(
            next_positions=labels.to(device),
            position_sequence_noise=sampled_noise.to(device),
            position_sequence=position.to(device),
            nparticles_per_example=n_particles_per_example.to(device),
            particle_types=particle_type.to(device),
            material_property=material_property.to(rank) if n_features == 3 else None
          )

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
        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
        for param in optimizer.param_groups:
          param['lr'] = lr_new

        if rank == 0 or device == torch.device("cpu"):
          print(f'Training step: {step}/{flags["ntraining_steps"]}. Loss: {loss}.')
          # Save model state
          if step % flags["nsave_steps"] == 0:
            if device == torch.device("cpu"):
              simulator.save(flags["model_path"] + 'model-'+str(step)+'.pt')
            else:
              simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
            train_state = dict(optimizer_state=optimizer.state_dict(),
                               global_train_state={"step": step},
                               loss=loss.item())
            torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

        # Complete training
        if (step >= flags["ntraining_steps"]):
          not_reached_nsteps = False
          break

        step += 1

  except KeyboardInterrupt:
    pass

  if rank == 0 or device == torch.device("cpu"):
    if device == torch.device("cpu"):
      simulator.save(flags["model_path"] + 'model-'+str(step)+'.pt')
    else:
      simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
    train_state = dict(optimizer_state=optimizer.state_dict(),
                       global_train_state={"step": step},
                       loss=loss.item())
    torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

  if torch.cuda.is_available():
    distribute.cleanup()


def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: torch.device) -> learned_simulator.LearnedSimulator:
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

  # Get necessary parameters for loading simulator.
  if "nnode_in" in metadata and "nedge_in" in metadata:
    nnode_in = metadata['nnode_in']
    nedge_in = metadata['nedge_in']
  else:
    # Given that there is no additional node feature (e.g., material_property) except for:
    # (position (dim), velocity (dim*6), particle_type (16)),
    nnode_in = 37 if metadata['dim'] == 3 else 30
    nedge_in = metadata['dim'] + 1

  # Init simulator.
  simulator = learned_simulator.LearnedSimulator(
      particle_dimensions=metadata['dim'],
      nnode_in=nnode_in,
      nedge_in=nedge_in,
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      boundary_clamp_limit=metadata["boundary_augment"] if "boundary_augment" in metadata else 1.0,
      device=device)

  return simulator


def main(_):
  """Train or evaluates the model.

  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device == torch.device('cuda'):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

  myflags = {}
  myflags["data_path"] = FLAGS.data_path
  myflags["noise_std"] = FLAGS.noise_std
  myflags["lr_init"] = FLAGS.lr_init
  myflags["lr_decay"] = FLAGS.lr_decay
  myflags["lr_decay_steps"] = FLAGS.lr_decay_steps
  myflags["batch_size"] = FLAGS.batch_size
  myflags["ntraining_steps"] = FLAGS.ntraining_steps
  myflags["nsave_steps"] = FLAGS.nsave_steps
  myflags["model_file"] = FLAGS.model_file
  myflags["model_path"] = FLAGS.model_path
  myflags["train_state_file"] = FLAGS.train_state_file

  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path)

    # Train on gpu 
    if device == torch.device('cuda'):
      world_size = torch.cuda.device_count()
      print(f"world_size = {world_size}")
      distribute.spawn_train(train, myflags, world_size, device)

    # Train on cpu  
    else:
      rank = None
      world_size = 1
      train(rank, myflags, world_size, device)

  elif FLAGS.mode in ['valid', 'rollout']:
    # Set device
    world_size = torch.cuda.device_count()
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
      device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
    predict(device)


if __name__ == '__main__':
  app.run(main)
