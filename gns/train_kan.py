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

import torch
import torch.distributed as dist
import torchvision.models as models
from torch.utils import collect_env
from torch.utils.data.distributed import DistributedSampler

import datetime
import time

from typing import List

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
flags.DEFINE_integer('validation_interval', None, help='Validation interval. Set `None` if validation loss is not needed')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')

flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")

# Argument for multinode training
flags.DEFINE_integer("local-rank", 0, help='local rank for distributed training')

# Parameters for KAN
flags.DEFINE_integer("use_kan", 0, help='set to 1 to use KAN, 0 to use MLP (default)')
flags.DEFINE_integer('kan_hidden_dim', 0, help="Latent dimension for KAN")

FLAGS = flags.FLAGS

if 'LOCAL_RANK' in os.environ:
  local_rank = int(os.environ['LOCAL_RANK'])



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


def predict(device: str, flags):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
  # Read metadata
  metadata = reading_utils.read_metadata(FLAGS.data_path, "rollout")
  # Params for KAN
  use_kan =flags['use_kan']
  kan_hidden_dim = flags['kan_hidden_dim']
  simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device, use_kan, kan_hidden_dim)

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
  split = 'test' if (FLAGS.mode == 'rollout' or (not os.path.isfile("{FLAGS.data_path}valid.npz"))) else 'valid'

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
      print(f"processing example number {example_i}")
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

def acceleration_loss(pred_acc, target_acc, non_kinematic_mask):
  """
  Compute the loss between predicted and target accelerations.

  Args:
    pred_acc: Predicted accelerations.
    target_acc: Target accelerations.
    non_kinematic_mask: Mask for kinematic particles.
  """
  loss = (pred_acc - target_acc) ** 2
  loss = loss.sum(dim=-1)
  num_non_kinematic = non_kinematic_mask.sum()
  loss = torch.where(non_kinematic_mask.bool(),
                    loss, torch.zeros_like(loss))
  loss = loss.sum() / num_non_kinematic
  return loss

def save_model_and_train_state(verbose, device, simulator, flags, step, epoch, optimizer,
                                train_loss, valid_loss, train_loss_hist, valid_loss_hist):
  """Save model state
  
  Args:
    verbose: is main rank or cpu
    device: torch device type
    simulator: Trained simulator if not will undergo training.
    flags: flags
    step: step
    epoch: epoch
    optimizer: optimizer
    train_loss: training loss at current step
    valid_loss: validation loss at current step
    train_loss_hist: training loss history at each epoch
    valid_loss_hist: validation loss history at each epoch
  """
  if verbose:
    if device == torch.device("cpu"):
        simulator.save(flags["model_path"] + 'model-' + str(step) + '.pt')
    else:
        simulator.module.save(flags["model_path"] + 'model-' + str(step) + '.pt')

    train_state = dict(optimizer_state=optimizer.state_dict(),
                        global_train_state={
                          "step": step, 
                          "epoch": epoch,
                          "train_loss": train_loss,
                          "valid_loss": valid_loss
                          },
                        loss_history={"train": train_loss_hist, "valid": valid_loss_hist}
                        )
    torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

def train(rank, flags, world_size, device, verbose):
  """Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
    device: torch device type
    verbose: gloabl rank 0 or cpu
  """
  if device == torch.device("cuda"):
    device_id = rank
  else:
    device_id = device

  # Read metadata
  metadata = reading_utils.read_metadata(flags["data_path"], "train")

  # Params for KAN
  use_kan =flags['use_kan']
  kan_hidden_dim = flags['kan_hidden_dim']

  # Get simulator and optimizer
  if device == torch.device("cuda"):
    serial_simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], rank, use_kan, kan_hidden_dim)
    serial_simulator = serial_simulator.to('cuda')
    device_id = rank
    simulator = DDP(serial_simulator, device_ids=[rank])
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"]*world_size)

  else:
    simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], device, use_kan, kan_hidden_dim)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"] * world_size)
 
  # Initialize training state
  step = 0
  epoch = 0
  steps_per_epoch = 0

  valid_loss = None
  epoch_train_loss = 0
  epoch_valid_loss = None

  train_loss_hist = []
  valid_loss_hist = []

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
      step = train_state["global_train_state"]["step"]
      epoch = train_state["global_train_state"]["epoch"]
      train_loss_hist = train_state["loss_history"]["train"]
      valid_loss_hist = train_state["loss_history"]["valid"]

    else:
      msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
      raise FileNotFoundError(msg)


  simulator.train()
  simulator.to(device_id)

  # Get data loader
  if device == torch.device("cuda"):
    path=f'{flags["data_path"]}train.npz'
    input_length_sequence=INPUT_SEQUENCE_LENGTH
    batch_size=flags["batch_size"]
    dataset = data_loader.SamplesDataset(path, input_length_sequence)
    # for multi node training we need to use pytorch's distributed sampler, see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dl = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size, pin_memory=True, collate_fn=data_loader.collate_fn)
  else:
    dl = data_loader.get_data_loader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                batch_size=flags["batch_size"])
  n_features = len(dl.dataset._data[0])

  # Load validation data
  if flags["validation_interval"] is not None:

    if device == torch.device("cuda"):
      path=f'{flags["data_path"]}valid.npz'
      input_length_sequence=INPUT_SEQUENCE_LENGTH
      batch_size=flags["batch_size"]
      dataset = data_loader.SamplesDataset(path, input_length_sequence)
      sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
      dl_valid = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size, pin_memory=True, collate_fn=data_loader.collate_fn)
    else:
      dl_valid = data_loader.get_data_loader_by_samples(path=f'{flags["data_path"]}valid.npz',
                                                  input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                  batch_size=flags["batch_size"])

      if len(dl_valid.dataset._data[0]) != n_features:
          raise ValueError(
              f"`n_features` of `valid.npz` and `train.npz` should be the same"
          )
      
  print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")

  start = time.time()
  try:
    while step < flags["ntraining_steps"]:
      if device == torch.device("cuda"):
        torch.distributed.barrier()
      else:
        pass
      for example in dl:  
        torch.cuda.empty_cache()
        steps_per_epoch += 1
        # ((position, particle_type, material_property, n_particles_per_example), labels) are in dl
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
        device_or_rank = rank if device == torch.device("cuda") else device
        pred_acc, target_acc = (simulator.module.predict_accelerations if device == torch.device("cuda") else simulator.predict_accelerations)(
            next_positions=labels.to(device_or_rank),
            position_sequence_noise=sampled_noise.to(device_or_rank),
            position_sequence=position.to(device_or_rank),
            nparticles_per_example=n_particles_per_example.to(device_or_rank),
            particle_types=particle_type.to(device_or_rank),
            material_property=material_property.to(device_or_rank) if n_features == 3 else None
        )
        
        # Validation
        if flags["validation_interval"] is not None:
          sampled_valid_example = next(iter(dl_valid))
          if step > 0 and step % flags["validation_interval"] == 0:
              valid_loss = validation(
                simulator, sampled_valid_example, n_features, flags, rank, device_id)
              print(f"Validation loss at {step}: {valid_loss.item()}")

        # Calculate the loss and mask out loss on kinematic particles
        loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

        train_loss = loss.item()
        epoch_train_loss += train_loss


        # Computes the gradient of loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
        for param in optimizer.param_groups:
          param['lr'] = lr_new

        if verbose:
          print(f'Training step: {step}/{flags["ntraining_steps"]}. Loss: {loss}.',flush=True)
          if step % 1000 == 0:
            print( '\nTraining time: {}'.format(
                  datetime.timedelta(seconds=time.time() - start),
              ),
              )
            start = time.time()
          # Save model state
          if step % flags["nsave_steps"] == 0:
            save_model_and_train_state(verbose, device, simulator, flags, step, epoch, \
                                       optimizer, train_loss, valid_loss, train_loss_hist, valid_loss_hist)

        step += 1
        if step >= flags["ntraining_steps"]:
          break


      # Epoch level statistics
      # Training loss at epoch
      epoch_train_loss /= steps_per_epoch
      epoch_train_loss = torch.tensor([epoch_train_loss]).to(device_id)
      if device == torch.device("cuda"):
        torch.distributed.reduce(epoch_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
        epoch_train_loss /= world_size

      train_loss_hist.append((epoch, epoch_train_loss.item()))

      # Validation loss at epoch
      if flags["validation_interval"] is not None:
        sampled_valid_example = next(iter(dl_valid))
        epoch_valid_loss = validation(
                simulator, sampled_valid_example, n_features, flags, rank, device_id)
        if device == torch.device("cuda"):
          torch.distributed.reduce(epoch_valid_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
          epoch_valid_loss /= world_size

        valid_loss_hist.append((epoch, epoch_valid_loss.item()))

      # Print epoch statistics
      if rank == 0 or device == torch.device("cpu"):
        print(f'Epoch {epoch}, training loss: {epoch_train_loss.item()}')
        if flags["validation_interval"] is not None:
          print(f'Epoch {epoch}, validation loss: {epoch_valid_loss.item()}')
      
      # Reset epoch training loss
      epoch_train_loss = 0
      if steps_per_epoch >= len(dl):
        epoch += 1
      steps_per_epoch = 0
      
      if step >= flags["ntraining_steps"]:
        break

  except KeyboardInterrupt:
    pass

  # Save model state on keyboard interrupt
  save_model_and_train_state(verbose, device, simulator, flags, step, epoch, optimizer, train_loss, valid_loss, train_loss_hist, valid_loss_hist)


  if torch.cuda.is_available():
    torch.distributed.destroy_process_group()


def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: torch.device,
        use_kan: int,
        kan_hidden_dim: int,
        ) -> learned_simulator.LearnedSimulator:
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
   # debug, TODO: change nmessage_passing_steps back to 10??
  simulator = learned_simulator.LearnedSimulator(
      particle_dimensions=metadata['dim'],
      nnode_in=nnode_in,
      nedge_in=nedge_in,
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=1,
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      boundary_clamp_limit=metadata["boundary_augment"] if "boundary_augment" in metadata else 1.0,
      device=device,
      use_kan=use_kan,
      kan_hidden_dim=kan_hidden_dim)

  return simulator

def validation(
        simulator,
        example,
        n_features,
        flags,
        rank,
        device_id):

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

  # Sample the noise to add to the inputs.
  sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
    position, noise_std_last_step=flags["noise_std"]).to(device_id)
  non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device_id)
  sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

  # Do evaluation for the validation data
  device_or_rank = rank if isinstance(device_id, int) else device_id
  # Select the appropriate prediction function
  predict_accelerations = simulator.module.predict_accelerations if isinstance(device_id, int) else simulator.predict_accelerations
  # Get the predictions and target accelerations
  with torch.no_grad():
      pred_acc, target_acc = predict_accelerations(
          next_positions=labels.to(device_or_rank),
          position_sequence_noise=sampled_noise.to(device_or_rank),
          position_sequence=position.to(device_or_rank),
          nparticles_per_example=n_particles_per_example.to(device_or_rank),
          particle_types=particle_type.to(device_or_rank),
          material_property=material_property.to(device_or_rank) if n_features == 3 else None
      )

  # Compute loss
  loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

  return loss

def main(_):
  """Train or evaluates the model.

  """
  torch.cuda.empty_cache()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  myflags = reading_utils.flags_to_dict(FLAGS)

  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path,exist_ok=True)

    # Train on gpu 
    if device == torch.device('cuda'):
      # available_gpus = torch.cuda.device_count()
      # print(f"Available GPUs = {available_gpus}")

      # # Set the number of GPUs based on availability and the specified number
      # if FLAGS.n_gpus is None or FLAGS.n_gpus > available_gpus:
      #   world_size = available_gpus
      #   if FLAGS.n_gpus is not None:
      #     print(f"Warning: The number of GPUs specified ({FLAGS.n_gpus}) exceeds the available GPUs ({available_gpus})")
      # else:
      #   world_size = FLAGS.n_gpus

      # # Print the status of GPU usage
      # print(f"Using {world_size}/{available_gpus} GPUs")

      # # Spawn training to GPUs
      # distribute.spawn_train(train, myflags, world_size, device)
      torch.multiprocessing.set_start_method('spawn')
      torch.distributed.init_process_group(
          backend="nccl",
          init_method='env://',
      )
      world_size = dist.get_world_size()
      torch.cuda.set_device(local_rank)
      torch.cuda.manual_seed(0)
      flags.DEFINE_boolean('verbose', False, 'Verbose.')
      FLAGS.verbose = dist.get_rank() == 0

      if FLAGS.verbose:
          print('Collecting env info...')
          print(collect_env.get_pretty_env_info())
          print()

      for r in range(torch.distributed.get_world_size()):
        if r == torch.distributed.get_rank():
            print(
                f'Global rank {torch.distributed.get_rank()} initialized: '
                f'local_rank = {local_rank}, '
                f'world_size = {torch.distributed.get_world_size()}',
            )
      
      train(local_rank, myflags, world_size, device, FLAGS.verbose)

    # Train on cpu  
    else:
      rank = None
      world_size = 1
      train(rank, myflags, world_size, device, True)

  elif FLAGS.mode in ['valid', 'rollout']:
    # Set device
    world_size = torch.cuda.device_count()
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
      device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
    #test code
    print(f"device is {device} world size is {world_size}")
    predict(device, myflags)


if __name__ == '__main__':
  app.run(main)
