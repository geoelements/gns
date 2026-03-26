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

from physicsnemo.distributed import (
    DistributedManager,
    mark_module_as_shared,
    ProcessGroupConfig,
    ProcessGroupNode,
)

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
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

# Parameter for MLP
flags.DEFINE_integer('mlp_hidden_dim', 128, help="Latent dimension for MLP")

# Parameters for model
flags.DEFINE_integer('latent_dim', 128, help="Latent dimension for model")
flags.DEFINE_integer('nmlp_layers', 1, help="MLP layers")

flags.DEFINE_boolean('verbose', False, 'Verbose.')

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
        device: torch.device,
        partition_group_name: str):
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
        partition_group_name=partition_group_name,
        material_property=material_property,
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


def predict(dist_manager, flags, partition_group_name):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
  # Read metadata
  metadata = reading_utils.read_metadata(FLAGS.data_path, "rollout")
  mlp_hidden_dim = flags['mlp_hidden_dim']
  simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, dist_manager).to(dist_manager.device)

  # Load simulator
  if os.path.exists(FLAGS.model_path + FLAGS.model_file):
    simulator.load(FLAGS.model_path + FLAGS.model_file)
  else:
    raise Exception(f"Model does not exist at {FLAGS.model_path + FLAGS.model_file}")

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
      positions = features[0].to(dist_manager.device)
      if metadata['sequence_length'] is not None:
        # If `sequence_length` is predefined in metadata,
        nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
      else:
        # If no predefined `sequence_length`, then get the sequence length
        sequence_length = positions.shape[1]
        nsteps = sequence_length - INPUT_SEQUENCE_LENGTH
      particle_type = features[1].to(dist_manager.device)
      if material_property_as_feature:
        material_property = features[2].to(dist_manager.device)
        n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(dist_manager.device)
      else:
        material_property = None
        n_particles_per_example = torch.tensor([int(features[2])], dtype=torch.int32).to(dist_manager.device)

      # Predict example rollout
      example_rollout, loss = rollout(simulator,
                                      positions,
                                      particle_type,
                                      material_property,
                                      n_particles_per_example,
                                      nsteps,
                                      dist_manager.device,
                                      partition_group_name,)

      example_rollout['metadata'] = metadata
      if flags.verbose:
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
  if flags.verbose:
    print("Mean loss on rollout prediction: {}".format(
      torch.mean(torch.cat(eval_loss))))


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
                                train_loss, valid_loss, train_loss_hist, valid_loss_hist,
                                epoch_train_loss):
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
    base_simulator = getattr(simulator, "module", simulator)
    base_simulator.save(flags["model_path"] + 'model-' + str(step) + '.mdlus')

    train_state = dict(optimizer_state=optimizer.state_dict(),
                        global_train_state={
                          "step": step, 
                          "epoch": epoch,
                          "train_loss": train_loss,
                          "valid_loss": valid_loss,
                          "epoch_train_loss": epoch_train_loss,
                          },
                        loss_history={"train": train_loss_hist, "valid": valid_loss_hist}
                        )
    torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

def train(dist_manager, graph_partition_pg_name, flags, verbose):
  """Train the model.

  Args:
    dist_manager: DistributedManager
    verbose: gloabl rank 0 or cpu
  """
  # Read metadata
  metadata = reading_utils.read_metadata(flags["data_path"], "train")

  # Get simulator and optimizer
  simulator = _get_simulator(
    metadata, flags["noise_std"], 
    flags["noise_std"], 
    dist_manager).to(dist_manager.device)
  if dist_manager.distributed and dist_manager.group_size("data_parallel") > 1:
    simulator = DDP(
      simulator, 
      process_group=dist_manager.group("data_parallel"),
      device_ids=[dist_manager.local_rank],
      output_device=dist_manager.device,
    )
  if (
    dist_manager.distributed
    and dist_manager.group_size(graph_partition_pg_name) > 1
  ):
    mark_module_as_shared(simulator, graph_partition_pg_name)
  try:
    dp_group_size = dist_manager.group_size("data_parallel")
  except:
    dp_group_size = 1
  optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"]*dp_group_size)

 
  # Initialize training state
  step = 0
  epoch = 0

  valid_loss = None
  epoch_train_loss = 0
  epoch_valid_loss = None

  train_loss_hist = []
  valid_loss_hist = []
  if (
    dist_manager.distributed
    and dist_manager.group_size(graph_partition_pg_name) > 1
  ):
    mp_rank = (dist_manager.group_rank("model_parallel"))
  else:
    mp_rank = 0
  if dist_manager.distributed and dist_manager.group_size("data_parallel") > 1:
    dp_rank = dist_manager.group_rank("data_parallel")
  else:
    dp_rank = 0

  # If model_path does exist and model_file and train_state_file exist continue training.
  if flags["model_file"] is not None:
    simulator, optimizer, step, epoch, train_loss_hist, valid_loss_hist, epoch_train_loss = _get_checkpoint(simulator, dist_manager, flags)

  simulator.train()

  # Get data loader
  path=f'{flags["data_path"]}train.npz'
  input_length_sequence=INPUT_SEQUENCE_LENGTH
  dl = data_loader.get_data_loader_by_samples(
          path,
          input_length_sequence,
          dist_manager,
          dp_rank,
          )
  n_features = len(dl.dataset._data[0])

  # Load validation data
  if flags["validation_interval"] is not None:

    path=f'{flags["data_path"]}valid.npz'
    input_length_sequence=INPUT_SEQUENCE_LENGTH
    dl_valid = data_loader.get_data_loader_by_samples(
          path,
          input_length_sequence,
          dist_manager,
          dp_rank,
          )
    if len(dl_valid.dataset._data[0]) != n_features:
      raise ValueError(
        f"`n_features` of `valid.npz` and `train.npz` should be the same"
      )
      
  start = time.time()
  try:
    while step < flags["ntraining_steps"]:
      torch.distributed.barrier()
      cur_step = step % len(dl)
      for example in dl: 
        # ((position, particle_type, material_property, n_particles_per_example), labels) are in dl
        position = example[0][0].to(dist_manager.device)
        particle_type = example[0][1].to(dist_manager.device)
        if n_features == 3:  # if dl includes material_property
          material_property = example[0][2].to(dist_manager.device)
          n_particles_per_example = example[0][3].to(dist_manager.device)
        elif n_features == 2:
          n_particles_per_example = example[0][2].to(dist_manager.device)
        else:
          raise NotImplementedError
        labels = example[1].to(dist_manager.device)
        n_particles_per_example.to(dist_manager.device)
        labels.to(dist_manager.device)
        # Sample the noise to add to the inputs to the model during training.
        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position, 
                noise_std_last_step=flags["noise_std"]).to(dist_manager.device)
        non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(dist_manager.device)
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

        # Get the predictions and target acceleration
        base_simulator = getattr(simulator, "module", simulator)
        
        pred_acc, target_acc = base_simulator.predict_accelerations(
            next_positions=labels.to(dist_manager.device),
            position_sequence_noise=sampled_noise.to(dist_manager.device),
            position_sequence=position.to(dist_manager.device),
            nparticles_per_example=n_particles_per_example.to(dist_manager.device),
            particle_types=particle_type.to(dist_manager.device),
            partition_group_name = graph_partition_pg_name,
            material_property=material_property.to(dist_manager.device) if n_features == 3 else None
        )
        # Validation
        if (
          flags["validation_interval"] is not None 
          and step > 0 
          and step % flags["validation_interval"] == 0 
        ):
          sampled_valid_example = next(iter(dl_valid))
          valid_loss = validation(
            simulator, sampled_valid_example, n_features, flags, dist_manager, graph_partition_pg_name)
          if verbose:
            print(f"Validation loss at {step}: {valid_loss.item()}")

        # Calculate the loss and mask out loss on kinematic particles
        loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)
        train_loss = loss.detach().item()
        epoch_train_loss += train_loss
        # Computes the gradient of loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * dp_group_size 
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

        step += 1
        cur_step += 1
        # Save model state
        if verbose and step % flags["nsave_steps"] == 0:
          save_model_and_train_state(verbose, torch.device("cuda"), \
            simulator, flags, step, epoch, \
            optimizer, train_loss, valid_loss, train_loss_hist, \
            valid_loss_hist, epoch_train_loss)
        if step >= flags["ntraining_steps"]:
          break
        if cur_step % len(dl) == 0:
          break

      # Epoch level statistics
      # Training loss at epoch
      step_this_epoch = len(dl) if (step != flags["ntraining_steps"]) else (step % len(dl))
      if step_this_epoch == 0:
        step_this_epoch = len(dl)
      epoch_train_loss /= step_this_epoch
      if verbose: 
        train_loss_hist.append((epoch, epoch_train_loss))

      # Validation loss at epoch
      if flags["validation_interval"] is not None:
        sampled_valid_example = next(iter(dl_valid))
        epoch_valid_loss = validation(
                simulator, 
                sampled_valid_example, 
                n_features, flags,
                dist_manager,
                graph_partition_pg_name,)
        if verbose:
          valid_loss_hist.append((epoch, epoch_valid_loss))

      # Print epoch statistics
      if verbose:
        print(f'Epoch {epoch}, training loss: {epoch_train_loss}')
        if flags["validation_interval"] is not None:
          print(f'Epoch {epoch}, validation loss: {epoch_valid_loss}')
      
      # Reset epoch training loss
      epoch_train_loss = 0
      if cur_step >= len(dl):
        epoch += 1
      cur_step = 0
      
      if step >= flags["ntraining_steps"]:
        break

  except KeyboardInterrupt:
    pass

  # Save model state on keyboard interrupt
  save_model_and_train_state(verbose, torch.device("cuda"), \
    simulator, flags, step, epoch, \
    optimizer, train_loss, valid_loss, train_loss_hist, valid_loss_hist, \
    epoch_train_loss)

  
  with open("log/valid_loss_hist.txt", "w") as f:
    for epoch, loss in valid_loss_hist:
      f.write(f"{loss}\n")


  if DistributedManager().distributed:
    DistributedManager.cleanup()

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

def _get_checkpoint(simulator, dist_manager, flags):
  if flags["model_file"] == "latest" and flags["train_state_file"] == "latest":
    # find the latest model, assumes model and train_state files are in step.
    fnames = glob.glob(f'{flags["model_path"]}*model*mdlus')
    max_model_number = 0
    expr = re.compile(r".*model-(\d+).mdlus")
    for fname in fnames:
      model_num = int(expr.search(fname).groups()[0])
      if model_num > max_model_number:
        max_model_number = model_num
    # reset names to point to the latest.
    flags["model_file"] = f"model-{max_model_number}.mdlus"
    flags["train_state_file"] = f"train_state-{max_model_number}.pt"

  if os.path.exists(flags["model_path"] + flags["model_file"]) and os.path.exists(flags["model_path"] + flags["train_state_file"]):
    # load model
    base_simulator = getattr(simulator, "module", simulator)
    base_simulator.load(flags["model_path"] + flags["model_file"], map_location=dist_manager.device)

    # load train state
    train_state = torch.load(flags["model_path"] + flags["train_state_file"])
    # set optimizer state
    optimizer = torch.optim.Adam(
        base_simulator.parameters())
    optimizer.load_state_dict(train_state["optimizer_state"])
    optimizer_to(optimizer, dist_manager.device)
    # set global train state
    step = train_state["global_train_state"]["step"]
    epoch = train_state["global_train_state"]["epoch"]
    train_loss_hist = train_state["loss_history"]["train"]
    valid_loss_hist = train_state["loss_history"]["valid"]
    epoch_train_loss = train_state["global_train_state"]["epoch_train_loss"]
  else:
    msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
    raise FileNotFoundError(msg)
  
  return simulator, optimizer, step, epoch, train_loss_hist, valid_loss_hist, epoch_train_loss

def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        dist_manager: DistributedManager,
        ) -> learned_simulator.LearnedSimulator:
  """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    acc_noise_std: Acceleration noise std deviation.
    vel_noise_std: Velocity noise std deviation.
  """

  # Normalization stats
  normalization_stats = {
      'acceleration': {
          'mean': metadata['acc_mean'], 
          'std': metadata['acc_std'], 
          'noise': acc_noise_std, 
      },
      'velocity': {
          'mean': metadata['vel_mean'],
          'std': metadata['vel_std'], 
          'noise': vel_noise_std,
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
      latent_dim=FLAGS.latent_dim,
      nmessage_passing_steps=10,
      nmlp_layers=FLAGS.nmlp_layers,
      mlp_hidden_dim=FLAGS.mlp_hidden_dim,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']).tolist(),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      boundary_clamp_limit=metadata["boundary_augment"] if "boundary_augment" in metadata else 1.0,
      device=str(dist_manager.device),
    )
  return simulator

def validation(
        simulator,
        example,
        n_features,
        flags,
        dist_manager,
        partition_group_name,
        ):

  position = example[0][0].to(dist_manager.device)
  particle_type = example[0][1].to(dist_manager.device)
  if n_features == 3:  # if dl includes material_property
    material_property = example[0][2].to(dist_manager.device)
    n_particles_per_example = example[0][3].to(dist_manager.device)
  elif n_features == 2:
    n_particles_per_example = example[0][2].to(dist_manager.device)
  else:
    raise NotImplementedError
  labels = example[1].to(dist_manager.device)

  # Sample the noise to add to the inputs.
  sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
    position, noise_std_last_step=flags["noise_std"]).to(dist_manager.device)
  non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(dist_manager.device)
  sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

  # Do evaluation for the validation data
  base_simulator = getattr(simulator, "module", simulator)
  predict_accelerations = base_simulator.predict_accelerations
  # Get the predictions and target accelerations
  with torch.no_grad():
      pred_acc, target_acc = predict_accelerations(
          next_positions=labels.to(dist_manager.device),
          position_sequence_noise=sampled_noise.to(dist_manager.device),
          position_sequence=position.to(dist_manager.device),
          nparticles_per_example=n_particles_per_example.to(dist_manager.device),
          particle_types=particle_type.to(dist_manager.device),
          material_property=material_property.to(dist_manager.device) if n_features == 3 else None,
          partition_group_name = partition_group_name,
      )

  # Compute loss
  loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

  return loss

def main(_):
  """Train or evaluates the model.

  """
  seed = 0
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  
  myflags = reading_utils.flags_to_dict(FLAGS)

  DistributedManager.initialize()
  if DistributedManager().distributed:
    graph_partition_pg_name = "model_parallel"
    world_size = torch.distributed.get_world_size()
    # debug: change graph partition size so there is ddp
    graph_partition_size = 3
    if not world_size % graph_partition_size == 0:
      raise ValueError(
          f"Partition Size ({graph_partition_size}) must divide World Size ({world_size}) evenly."
            )
    world = ProcessGroupNode("world")
    pg_config = ProcessGroupConfig(world)
    pg_config.add_node(ProcessGroupNode("data_parallel"), parent=world)
    pg_config.add_node(ProcessGroupNode("model_parallel"), parent=world)
    pg_sizes = {
      "model_parallel": graph_partition_size,
      "data_parallel": world_size // graph_partition_size,
    }
    pg_config.set_leaf_group_sizes(pg_sizes)
    DistributedManager.create_groups_from_config(
      pg_config,
      verbose=False,
    )
  else:
    world_size = 1
    graph_partition_size = 1
    graph_partition_pg_name = None
  dist_manager = DistributedManager()
  FLAGS.verbose = dist_manager.rank == 0

  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path,exist_ok=True)

    train(dist_manager, graph_partition_pg_name, myflags, FLAGS.verbose)


  elif FLAGS.mode in ['valid', 'rollout']:
    predict(dist_manager, myflags, graph_partition_pg_name)

  


if __name__ == '__main__':
  app.run(main)
