import collections
import json
import os
import pickle
import glob
import re
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute
from gns.args import Config

Stats = collections.namedtuple("Stats", ["mean", "std"])


def rollout(
    simulator: learned_simulator.LearnedSimulator,
    cfg: DictConfig,
    position: torch.tensor,
    particle_types: torch.tensor,
    material_property: torch.tensor,
    n_particles_per_example: torch.tensor,
    nsteps: int,
    device: torch.device,
):
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

    initial_positions = position[:, : cfg.data.input_sequence_length]
    ground_truth_positions = position[:, cfg.data.input_sequence_length :]

    current_positions = initial_positions
    predictions = []

    for step in tqdm(range(nsteps), total=nsteps):
        # Get next position with shape (nnodes, dim)
        next_position = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_types,
            material_property=material_property,
        )

        # Update kinematic particles from prescribed trajectory.
        kinematic_mask = (
            (particle_types == cfg.data.kinematic_particle_id)
            .clone()
            .detach()
            .to(device)
        )
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(
            -1, current_positions.shape[-1]
        )
        next_position = torch.where(
            kinematic_mask, next_position_ground_truth, next_position
        )
        predictions.append(next_position)

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1
        )

    # Predictions with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

    loss = (predictions - ground_truth_positions) ** 2

    output_dict = {
        "initial_positions": initial_positions.permute(1, 0, 2).cpu().numpy(),
        "predicted_rollout": predictions.cpu().numpy(),
        "ground_truth_rollout": ground_truth_positions.cpu().numpy(),
        "particle_types": particle_types.cpu().numpy(),
        "material_property": (
            material_property.cpu().numpy() if material_property is not None else None
        ),
    }

    return output_dict, loss


def predict(device: str, cfg: DictConfig):
    """Predict rollouts.

    Args:
      device: 'cpu' or 'cuda'.
      cfg: configuration dictionary.

    """
    # Read metadata
    metadata = reading_utils.read_metadata(cfg.data.path, "rollout")
    simulator = _get_simulator(
        metadata,
        cfg.data.num_particle_types,
        cfg.data.noise_std,
        cfg.data.noise_std,
        device,
    )

    # Load simulator
    if os.path.exists(cfg.model.path + cfg.model.file):
        simulator.load(cfg.model.path + cfg.model.file)
    else:
        raise Exception(f"Model does not exist at {cfg.model.path + cfg.model.file}")

    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(cfg.output.path):
        os.makedirs(cfg.output.path)

    # Use `valid`` set for eval mode if not use `test`
    split = (
        "test"
        if (cfg.mode == "rollout" or (not os.path.isfile("{cfg.data.path}valid.npz")))
        else "valid"
    )

    # Get dataset
    ds = data_loader.get_data_loader_by_trajectories(path=f"{cfg.data.path}{split}.npz")
    # See if our dataset has material property as feature
    if (
        len(ds.dataset._data[0]) == 3
    ):  # `ds` has (positions, particle_type, material_property)
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
            if metadata["sequence_length"] is not None:
                # If `sequence_length` is predefined in metadata,
                nsteps = metadata["sequence_length"] - cfg.data.input_sequence_length
            else:
                # If no predefined `sequence_length`, then get the sequence length
                sequence_length = positions.shape[1]
                nsteps = sequence_length - cfg.data.input_sequence_length
            particle_type = features[1].to(device)
            if material_property_as_feature:
                material_property = features[2].to(device)
                n_particles_per_example = torch.tensor(
                    [int(features[3])], dtype=torch.int32
                ).to(device)
            else:
                material_property = None
                n_particles_per_example = torch.tensor(
                    [int(features[2])], dtype=torch.int32
                ).to(device)

            # Predict example rollout
            example_rollout, loss = rollout(
                simulator,
                cfg,
                positions,
                particle_type,
                material_property,
                n_particles_per_example,
                nsteps,
                device,
            )

            example_rollout["metadata"] = metadata
            print("Predicting example {} loss: {}".format(example_i, loss.mean()))
            eval_loss.append(torch.flatten(loss))

            # Save rollout in testing
            if cfg.mode == "rollout":
                example_rollout["metadata"] = metadata
                example_rollout["loss"] = loss.mean()
                filename = f"{cfg.output.filename}_ex{example_i}.pkl"
                filename = os.path.join(cfg.output.path, filename)
                with open(filename, "wb") as f:
                    pickle.dump(example_rollout, f)

    print(
        "Mean loss on rollout prediction: {}".format(torch.mean(torch.cat(eval_loss)))
    )


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
    loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
    loss = loss.sum() / num_non_kinematic
    return loss


def save_model_and_train_state(
    rank,
    device,
    simulator,
    cfg,
    step,
    epoch,
    optimizer,
    train_loss,
    valid_loss,
    train_loss_hist,
    valid_loss_hist,
):
    """Save model state

    Args:
      rank: local rank
      device: torch device type
      simulator: Trained simulator if not will undergo training.
      cfg: Configuration dictionary.
      step: step
      epoch: epoch
      optimizer: optimizer
      train_loss: training loss at current step
      valid_loss: validation loss at current step
      train_loss_hist: training loss history at each epoch
      valid_loss_hist: validation loss history at each epoch
    """
    if rank == 0 or device == torch.device("cpu"):
        if device == torch.device("cpu"):
            simulator.save(cfg.model.path + "model-" + str(step) + ".pt")
        else:
            simulator.module.save(cfg.model.path + "model-" + str(step) + ".pt")

        train_state = dict(
            optimizer_state=optimizer.state_dict(),
            global_train_state={
                "step": step,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            },
            loss_history={"train": train_loss_hist, "valid": valid_loss_hist},
        )
        torch.save(train_state, f"{cfg.model.path}train_state-{step}.pt")


def train(rank, cfg, world_size, device):
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
    metadata = reading_utils.read_metadata(cfg.data.path, "train")

    # Get simulator and optimizer
    if device == torch.device("cuda"):
        serial_simulator = _get_simulator(
            metadata,
            cfg.data.num_particle_types,
            cfg.data.noise_std,
            cfg.data.noise_std,
            rank,
        )
        simulator = DDP(
            serial_simulator.to(rank), device_ids=[rank], output_device=rank
        )
        optimizer = torch.optim.Adam(
            simulator.parameters(), lr=cfg.training.learning_rate.initial * world_size
        )
    else:
        simulator = _get_simulator(
            metadata,
            cfg.data.num_particle_types,
            cfg.data.noise_std,
            cfg.data.noise_std,
            device,
        )
        optimizer = torch.optim.Adam(
            simulator.parameters(), lr=cfg.training.learning_rate.initial * world_size
        )

    # Initialize training state
    step = 0
    epoch = 0
    steps_per_epoch = 0

    valid_loss = None
    train_loss = 0
    epoch_valid_loss = None

    train_loss_hist = []
    valid_loss_hist = []

    # If model_path does exist and model_file and train_state_file exist continue training.
    if cfg.model.file is not None and cfg.training.resume:
        if cfg.model.file == "latest" and cfg.model.train_state_file == "latest":
            # find the latest model, assumes model and train_state files are in step.
            fnames = glob.glob(f"{cfg.model.path}*model*pt")
            max_model_number = 0
            expr = re.compile(".*model-(\d+).pt")
            for fname in fnames:
                model_num = int(expr.search(fname).groups()[0])
                if model_num > max_model_number:
                    max_model_number = model_num
            # reset names to point to the latest.
            cfg.model.file = f"model-{max_model_number}.pt"
            cfg.model.train_state_file = f"train_state-{max_model_number}.pt"

        if os.path.exists(cfg.model.path + cfg.model.file) and os.path.exists(
            cfg.model.path + cfg.model.train_state_file
        ):
            # load model
            if device == torch.device("cuda"):
                simulator.module.load(cfg.model.path + cfg.model.file)
            else:
                simulator.load(cfg.model.path + cfg.model.file)

            # load train state
            train_state = torch.load(cfg.model.path + cfg.model.train_state_file)

            # set optimizer state
            optimizer = torch.optim.Adam(
                simulator.module.parameters()
                if device == torch.device("cuda")
                else simulator.parameters()
            )
            optimizer.load_state_dict(train_state["optimizer_state"])
            optimizer_to(optimizer, device_id)

            # set global train state
            step = train_state["global_train_state"]["step"]
            epoch = train_state["global_train_state"]["epoch"]
            train_loss_hist = train_state["loss_history"]["train"]
            valid_loss_hist = train_state["loss_history"]["valid"]

        else:
            msg = f"Specified model_file {cfg.model.path + cfg.model.file} and train_state_file {cfg.model.path + cfg.model.train_state_file} not found."
            raise FileNotFoundError(msg)

    simulator.train()
    simulator.to(device_id)

    # Get data loader
    get_data_loader = (
        distribute.get_data_distributed_dataloader_by_samples
        if device == torch.device("cuda")
        else data_loader.get_data_loader_by_samples
    )

    # Load training data
    dl = get_data_loader(
        path=f"{cfg.data.path}train.npz",
        input_length_sequence=cfg.data.input_sequence_length,
        batch_size=cfg.data.batch_size,
    )
    n_features = len(dl.dataset._data[0])

    # Load validation data
    if cfg.training.validation_interval is not None:
        dl_valid = get_data_loader(
            path=f"{cfg.data.path}valid.npz",
            input_length_sequence=cfg.data.input_sequence_length,
            batch_size=cfg.data.batch_size,
        )
        if len(dl_valid.dataset._data[0]) != n_features:
            raise ValueError(
                f"`n_features` of `valid.npz` and `train.npz` should be the same"
            )

    print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")

    if rank == 0 or device == torch.device("cpu"):
        writer = SummaryWriter(log_dir=cfg.logging.tensorboard_dir)

        writer.add_text("metadata", json.dumps(metadata, indent=4))
        yaml_config = OmegaConf.to_yaml(cfg)
        writer.add_text("Config", yaml_config, global_step=0)

        # Log hyperparameters
        hparam_dict = {
            "lr_init": cfg.training.learning_rate.initial,
            "lr_decay": cfg.training.learning_rate.decay,
            "lr_decay_steps": cfg.training.learning_rate.decay_steps,
            "batch_size": cfg.data.batch_size,
            "noise_std": cfg.data.noise_std,
            "ntraining_steps": cfg.training.steps,
        }
        metric_dict = {"train_loss": 0, "valid_loss": 0}  # Initial values
        writer.add_hparams(hparam_dict, metric_dict)

    try:
        num_epochs = max(
            1, (cfg.training.steps + len(dl) - 1) // len(dl)
        )  # Calculate total epochs
        print(f"Total epochs = {num_epochs}")
        for epoch in tqdm(range(epoch, num_epochs), desc="Training", unit="epoch"):
            if device == torch.device("cuda"):
                torch.distributed.barrier()

            epoch_loss = 0.0
            steps_this_epoch = 0

            # Create a tqdm progress bar for each epoch
            with tqdm(total=len(dl), desc=f"Epoch {epoch}", unit="batch") as pbar:
                for example in dl:
                    steps_per_epoch += 1
                    position = example[0][0].to(device_id)
                    particle_type = example[0][1].to(device_id)
                    if n_features == 3:
                        material_property = example[0][2].to(device_id)
                        n_particles_per_example = example[0][3].to(device_id)
                    elif n_features == 2:
                        n_particles_per_example = example[0][2].to(device_id)
                    else:
                        raise NotImplementedError
                    labels = example[1].to(device_id)

                    n_particles_per_example = n_particles_per_example.to(device_id)
                    labels = labels.to(device_id)

                    sampled_noise = (
                        noise_utils.get_random_walk_noise_for_position_sequence(
                            position, noise_std_last_step=cfg.data.noise_std
                        ).to(device_id)
                    )
                    non_kinematic_mask = (
                        (particle_type != cfg.data.kinematic_particle_id)
                        .clone()
                        .detach()
                        .to(device_id)
                    )
                    sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

                    device_or_rank = rank if device == torch.device("cuda") else device
                    predict_fn = (
                        simulator.module.predict_accelerations
                        if device == torch.device("cuda")
                        else simulator.predict_accelerations
                    )
                    pred_acc, target_acc = predict_fn(
                        next_positions=labels.to(device_or_rank),
                        position_sequence_noise=sampled_noise.to(device_or_rank),
                        position_sequence=position.to(device_or_rank),
                        nparticles_per_example=n_particles_per_example.to(
                            device_or_rank
                        ),
                        particle_types=particle_type.to(device_or_rank),
                        material_property=(
                            material_property.to(device_or_rank)
                            if n_features == 3
                            else None
                        ),
                    )

                    if (
                        cfg.training.validation_interval is not None
                        and step > 0
                        and step % cfg.training.validation_interval == 0
                    ):
                        if rank == 0 or device == torch.device("cpu"):
                            sampled_valid_example = next(iter(dl_valid))
                            valid_loss = validation(
                                simulator,
                                sampled_valid_example,
                                n_features,
                                cfg,
                                rank,
                                device_id,
                            )
                            writer.add_scalar("Loss/valid", valid_loss.item(), step)

                    loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

                    train_loss = loss.item()
                    epoch_loss += train_loss
                    steps_this_epoch += 1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    lr_new = (
                        cfg.training.learning_rate.initial
                        * (
                            cfg.training.learning_rate.decay
                            ** (step / cfg.training.learning_rate.decay_steps)
                        )
                        * world_size
                    )
                    for param in optimizer.param_groups:
                        param["lr"] = lr_new

                    # Log training loss
                    if rank == 0 or device == torch.device("cpu"):
                        writer.add_scalar("Loss/train", train_loss, step)
                        writer.add_scalar("Learning Rate", lr_new, step)

                    avg_loss = epoch_loss / steps_this_epoch
                    pbar.set_postfix(
                        loss=f"{train_loss:.2f}",
                        avg_loss=f"{avg_loss:.2f}",
                        lr=f"{lr_new:.2e}",
                    )
                    pbar.update(1)

                    if (
                        rank == 0 or device == torch.device("cpu")
                    ) and step % cfg.training.save_steps == 0:
                        save_model_and_train_state(
                            rank,
                            device,
                            simulator,
                            cfg,
                            step,
                            epoch,
                            optimizer,
                            train_loss,
                            valid_loss,
                            train_loss_hist,
                            valid_loss_hist,
                        )

                    step += 1
                    if step >= cfg.training.steps:
                        break

            # Epoch level statistics
            avg_loss = torch.tensor([epoch_loss / steps_this_epoch]).to(device_id)
            if device == torch.device("cuda"):
                torch.distributed.reduce(
                    avg_loss, dst=0, op=torch.distributed.ReduceOp.SUM
                )
                avg_loss /= world_size

            train_loss_hist.append((epoch, avg_loss.item()))

            if cfg.training.validation_interval is not None:
                sampled_valid_example = next(iter(dl_valid))
                epoch_valid_loss = validation(
                    simulator, sampled_valid_example, n_features, cfg, rank, device_id
                )
                if device == torch.device("cuda"):
                    torch.distributed.reduce(
                        epoch_valid_loss, dst=0, op=torch.distributed.ReduceOp.SUM
                    )
                    epoch_valid_loss /= world_size
                valid_loss_hist.append((epoch, epoch_valid_loss.item()))

            if rank == 0 or device == torch.device("cpu"):
                writer.add_scalar("Loss/train_epoch", avg_loss.item(), epoch)
                if cfg.training.validation_interval is not None:
                    writer.add_scalar(
                        "Loss/valid_epoch", epoch_valid_loss.item(), epoch
                    )

            if step >= cfg.training.steps:
                break
    except KeyboardInterrupt:
        pass

    # Save model state on keyboard interrupt
    save_model_and_train_state(
        rank,
        device,
        simulator,
        cfg,
        step,
        epoch,
        optimizer,
        train_loss,
        valid_loss,
        train_loss_hist,
        valid_loss_hist,
    )

    if rank == 0 or device == torch.device("cpu"):
        writer.close()

    if torch.cuda.is_available():
        distribute.cleanup()


def _get_simulator(
    metadata: json,
    num_particle_types: int,
    acc_noise_std: float,
    vel_noise_std: float,
    device: torch.device,
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
        "acceleration": {
            "mean": torch.FloatTensor(metadata["acc_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["acc_std"]) ** 2 + acc_noise_std**2
            ).to(device),
        },
        "velocity": {
            "mean": torch.FloatTensor(metadata["vel_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["vel_std"]) ** 2 + vel_noise_std**2
            ).to(device),
        },
    }

    # Get necessary parameters for loading simulator.
    if "nnode_in" in metadata and "nedge_in" in metadata:
        nnode_in = metadata["nnode_in"]
        nedge_in = metadata["nedge_in"]
    else:
        # Given that there is no additional node feature (e.g., material_property) except for:
        # (position (dim), velocity (dim*6), particle_type (16)),
        nnode_in = 37 if metadata["dim"] == 3 else 30
        nedge_in = metadata["dim"] + 1

    # Init simulator.
    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata["dim"],
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=128,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata["default_connectivity_radius"],
        boundaries=np.array(metadata["bounds"]),
        normalization_stats=normalization_stats,
        nparticle_types=num_particle_types,
        particle_type_embedding_size=16,
        boundary_clamp_limit=(
            metadata["boundary_augment"] if "boundary_augment" in metadata else 1.0
        ),
        device=device,
    )

    return simulator


def validation(simulator, example, n_features, cfg, rank, device_id):
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
        position, noise_std_last_step=cfg.data.noise_std
    ).to(device_id)
    non_kinematic_mask = (
        (particle_type != cfg.data.kinematic_particle_id).clone().detach().to(device_id)
    )
    sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

    # Do evaluation for the validation data
    device_or_rank = rank if isinstance(device_id, int) else device_id
    # Select the appropriate prediction function
    predict_accelerations = (
        simulator.module.predict_accelerations
        if isinstance(device_id, int)
        else simulator.predict_accelerations
    )
    # Get the predictions and target accelerations
    with torch.no_grad():
        pred_acc, target_acc = predict_accelerations(
            next_positions=labels.to(device_or_rank),
            position_sequence_noise=sampled_noise.to(device_or_rank),
            position_sequence=position.to(device_or_rank),
            nparticles_per_example=n_particles_per_example.to(device_or_rank),
            particle_types=particle_type.to(device_or_rank),
            material_property=(
                material_property.to(device_or_rank) if n_features == 3 else None
            ),
        )

    # Compute loss
    loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

    return loss


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: Config):
    """Train or evaluates the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    if cfg.mode == "train":
        # If model_path does not exist create new directory.
        if not os.path.exists(cfg.model.path):
            os.makedirs(cfg.model.path)

        # Create TensorBoard log directory
        if not os.path.exists(cfg.logging.tensorboard_dir):
            os.makedirs(cfg.logging.tensorboard_dir)

        # Train on gpu
        if device == torch.device("cuda"):
            available_gpus = torch.cuda.device_count()
            print(f"Available GPUs = {available_gpus}")

            # Set the number of GPUs based on availability and the specified number
            if cfg.hardware.n_gpus is None or cfg.hardware.n_gpus > available_gpus:
                world_size = available_gpus
                if cfg.hardware.n_gpus is not None:
                    print(
                        f"Warning: The number of GPUs specified ({cfg.hardware.n_gpus}) exceeds the available GPUs ({available_gpus})"
                    )
            else:
                world_size = cfg.hardware.n_gpus

            # Print the status of GPU usage
            print(f"Using {world_size}/{available_gpus} GPUs")

            # Spawn training to GPUs
            distribute.spawn_train(train, cfg, world_size, device)

        # Train on cpu
        else:
            rank = None
            world_size = 1
            train(rank, cfg, world_size, device)

    elif cfg.mode in ["valid", "rollout"]:
        # Set device
        world_size = torch.cuda.device_count()
        if cfg.hardware.cuda_device_number is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{int(cfg.hardware.cuda_device_number)}")
        # test code
        print(f"device is {device} world size is {world_size}")
        predict(device, cfg)


if __name__ == "__main__":
    main()
