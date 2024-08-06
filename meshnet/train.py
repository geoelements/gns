import sys
import os
import glob
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_geometric.transforms as T
import re
import pickle
from tqdm import tqdm
import json

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from meshnet import data_loader
from meshnet import learned_simulator
from meshnet.noise import get_velocity_noise
from meshnet.utils import datas_to_graph
from meshnet.utils import NodeType
from meshnet.utils import optimizer_to
from meshnet.args import Config
from meshnet import reading_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# an instance that transforms face-based graph to edge-based graph. Edge features are auto-computed using "Cartesian" and "Distance"
transformer = T.Compose(
    [T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)]
)


def setup_tensorboard(cfg, metadata):
    """Setup tensorboard.

    Args:
        cfg: Configuration dictionary.
        metadata: Metadata.
    """
    writer = SummaryWriter(log_dir=cfg.logging.tensorboard_dir)

    if metadata:
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
    return writer


def save_model_and_train_state(
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
    simulator.save(cfg.model.path + "model-" + str(step) + ".pt")
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


def predict(simulator: learned_simulator.MeshSimulator, device: str, cfg):
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
    split = "test" if cfg.mode == "rollout" else "valid"

    # Load trajectory data.
    ds = data_loader.get_data_loader_by_trajectories(path=f"{cfg.data.path}{split}.npz")

    # Rollout
    with torch.no_grad():
        for i, features in enumerate(ds):
            nsteps = len(features[0]) - cfg.data.input_sequence_length
            prediction_data = rollout(simulator, features, nsteps, device, cfg)
            print(f"Rollout for example{i}: loss = {prediction_data['mean_loss']}")

            # Save rollout in testing
            if cfg.mode == "rollout":
                filename = f"{cfg.output.filename}_{i}.pkl"
                filename = os.path.join(cfg.output.path, filename)
                with open(filename, "wb") as f:
                    pickle.dump(prediction_data, f)

    print(f"Mean loss on rollout prediction: {prediction_data['mean_loss']}")


def rollout(
    simulator: learned_simulator.MeshSimulator, features, nsteps: int, device, cfg
):
    node_coords = features[0]  # (timesteps, nnode, ndims)
    node_types = features[1]  # (timesteps, nnode, )
    velocities = features[2]  # (timesteps, nnode, ndims)
    pressures = features[3]  # (timesteps, nnode, )
    cells = features[4]  # # (timesteps, ncells, nnode_per_cell)

    initial_velocities = velocities[: cfg.data.input_sequence_length]
    ground_truth_velocities = velocities[cfg.data.input_sequence_length :]

    current_velocities = initial_velocities.squeeze().to(device)
    predictions = []
    mask = None

    for step in tqdm(range(nsteps), total=nsteps):
        # Predict next velocity
        # First, obtain data to form a graph
        current_node_coords = node_coords[step]
        current_node_type = node_types[step]
        current_pressure = pressures[step]
        current_cell = cells[step]
        current_time_idx_vector = (
            torch.tensor(np.full(current_node_coords.shape[0], step))
            .to(torch.float32)
            .contiguous()
        )
        next_ground_truth_velocities = ground_truth_velocities[step].to(device)
        current_example = (
            (
                current_node_coords,
                current_node_type,
                current_velocities,
                current_pressure,
                current_cell,
                current_time_idx_vector,
            ),
            next_ground_truth_velocities,
        )

        # Make graph
        graph = datas_to_graph(current_example, dt=cfg.data.dt, device=device)
        # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
        graph = transformer(graph)

        # Predict next velocity
        predicted_next_velocity = simulator.predict_velocity(
            current_velocities=graph.x[:, 1:3],
            node_type=graph.x[:, 0],
            edge_index=graph.edge_index,
            edge_features=graph.edge_attr,
        )

        # Apply mask.
        if (
            mask is None
        ):  # only compute mask for the first timestep, since it will be the same for the later timesteps
            mask = torch.logical_or(
                current_node_type == NodeType.NORMAL,
                current_node_type == NodeType.OUTFLOW,
            )
            mask = torch.logical_not(mask)
            mask = mask.squeeze(1)
        # Maintain previous velocity if node_type is not (Normal or Outflow).
        # i.e., only update normal or outflow nodes.
        predicted_next_velocity[mask] = next_ground_truth_velocities[mask]
        predictions.append(predicted_next_velocity)

        # Update current position for the next prediction
        current_velocities = predicted_next_velocity.to(device)

    # Prediction with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)

    loss = (predictions - ground_truth_velocities.to(device)) ** 2

    output_dict = {
        "initial_velocities": initial_velocities.cpu().numpy(),
        "predicted_rollout": predictions.cpu().numpy(),
        "ground_truth_rollout": ground_truth_velocities.cpu().numpy(),
        "node_coords": node_coords.cpu().numpy(),
        "node_types": node_types.cpu().numpy(),
        "mean_loss": loss.mean().cpu().numpy(),
    }

    return output_dict


def train(simulator, cfg):
    print(f"device = {device}")

    # missing metadata for the dataset
    # metadata = reading_utils.read_metadata(cfg.data.path, "train")
    metadata = None
    writer = setup_tensorboard(cfg, metadata)
    # Initiate training.
    optimizer = torch.optim.Adam(
        simulator.parameters(), lr=cfg.training.learning_rate.initial
    )
    step = 0
    epoch = 0

    valid_loss = None
    train_loss = 0
    epoch_valid_loss = None

    train_loss_hist = []
    valid_loss_hist = []

    # Set model and its path to save, and load model.
    # If model_path does not exist create new directory and begin training.
    model_path = cfg.model.path
    if not os.path.exists(cfg.model.path):
        os.makedirs(cfg.model.path, exist_ok=True)

    # Create TensorBoard log directory
    if not os.path.exists(cfg.logging.tensorboard_dir):
        os.makedirs(cfg.logging.tensorboard_dir)

    # If model_path does exist and model_file and train_state_file exist continue training.
    if cfg.model.file is not None and cfg.training.resume:
        if cfg.model.file == "latest" and cfg.model.train_state_file == "latest":
            # find the latest model, assumes model and train_state files are in step.
            fnames = glob.glob(f"{model_path}*model*pt")
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
            simulator.load(cfg.model.path + cfg.model.file)

            # load train state
            train_state = torch.load(cfg.model.path + cfg.model.train_state_file)
            # set optimizer state
            optimizer = torch.optim.Adam(simulator.parameters())
            optimizer.load_state_dict(train_state["optimizer_state"])
            optimizer_to(optimizer, device)
            # set global train state
            step = train_state["global_train_state"]["step"]
            epoch = train_state["global_train_state"]["epoch"]
            train_loss_hist = train_state["loss_history"]["train"]
            valid_loss_hist = train_state["loss_history"]["valid"]
        else:
            raise FileNotFoundError(
                f"Specified model_file {cfg.model.path + cfg.model.file} and train_state_file {cfg.model.path + cfg.model.train_state_file} not found."
            )

    simulator.train()
    simulator.to(device)

    # Load data
    ds = data_loader.get_data_loader_by_samples(
        path=f"{cfg.data.path}/{cfg.mode}.npz",
        input_length_sequence=cfg.data.input_sequence_length,
        dt=cfg.data.dt,
        batch_size=cfg.data.batch_size,
    )
    valid_dl = data_loader.get_data_loader_by_samples(
        path=f"{cfg.data.path}/valid.npz",
        input_length_sequence=cfg.data.input_sequence_length,
        dt=cfg.data.dt,
        batch_size=cfg.data.batch_size,
    )

    try:
        num_epochs = max(1, (cfg.training.steps + len(ds) - 1) // len(ds))
        print(f"Total epochs = {num_epochs}")
        for epoch in tqdm(
            range(epoch, num_epochs), desc="Training", unit="epoch", disable=False
        ):
            epoch_loss = 0.0
            steps_this_epoch = 0
            # Create a tqdm progress bar for each epoch
            with tqdm(
                # resume from one step after the checkpoint
                range(step % len(ds) + 1, len(ds)),
                desc=f"Epoch {epoch}",
                unit="batch",
                disable=False,
            ) as pbar:
                for i, graph in enumerate(ds):
                    # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
                    graph = transformer(graph.to(device))

                    # Get inputs
                    node_types = graph.x[:, 0]
                    current_velocities = graph.x[:, 1:3]
                    edge_index = graph.edge_index
                    edge_features = graph.edge_attr
                    target_velocities = graph.y

                    # Get velocity noise
                    velocity_noise = get_velocity_noise(
                        graph, noise_std=cfg.data.noise_std, device=device
                    )

                    # Predict dynamics
                    pred_acc, target_acc = simulator.predict_acceleration(
                        current_velocities=current_velocities,
                        node_type=node_types,
                        edge_index=edge_index,
                        edge_features=edge_features,
                        target_velocities=target_velocities,
                        velocity_noise=velocity_noise,
                    )

                    # Compute loss
                    mask = torch.logical_or(
                        node_types == NodeType.NORMAL, node_types == NodeType.OUTFLOW
                    )
                    errors = ((pred_acc - target_acc) ** 2)[
                        mask
                    ]  # only compute errors if node_types is NORMAL or OUTFLOW
                    loss = torch.mean(errors)

                    # Computes the gradient of loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update learning rate
                    lr_new = (
                        cfg.training.learning_rate.initial
                        * cfg.training.learning_rate.decay
                        ** (step / cfg.training.learning_rate.decay_steps)
                        + 1e-6
                    )
                    for param in optimizer.param_groups:
                        param["lr"] = lr_new

                    train_loss = loss.item()
                    epoch_loss += train_loss
                    steps_this_epoch += 1
                    avg_loss = epoch_loss / steps_this_epoch
                    writer.add_scalar("Loss/train", train_loss, step)
                    writer.add_scalar("Learning Rate", lr_new, step)
                    pbar.set_postfix(
                        loss=f"{train_loss:.2f}",
                        avg_loss=f"{avg_loss:.2f}",
                        lr=f"{lr_new:.2e}",
                    )
                    pbar.update(1)

                    if (
                        cfg.training.validation_interval is not None
                        and step > 0
                        and step % cfg.training.validation_interval == 0
                    ):

                        sampled_valid_example = next(iter(valid_dl))
                        valid_loss = validation(simulator, sampled_valid_example, cfg)
                        writer.add_scalar("Loss/valid", valid_loss.item(), step)

                    # Save model state
                    if step % cfg.training.save_steps == 0:
                        save_model_and_train_state(
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

            avg_loss = torch.tensor([epoch_loss / steps_this_epoch])
            train_loss_hist.append((epoch, avg_loss.item()))

            if cfg.training.validation_interval is not None:
                sampled_valid_example = next(iter(valid_dl))
                epoch_valid_loss = validation(
                    simulator,
                    sampled_valid_example,
                    cfg,
                )
                valid_loss_hist.append((epoch, epoch_valid_loss.item()))
                writer.add_scalar("Loss/valid_epoch", epoch_valid_loss.item(), epoch)

    except KeyboardInterrupt:
        pass

    save_model_and_train_state(
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


def validation(simulator, graph, cfg):

    graph = transformer(graph.to(device))
    # Get inputs
    node_types = graph.x[:, 0]
    current_velocities = graph.x[:, 1:3]
    edge_index = graph.edge_index
    edge_features = graph.edge_attr
    target_velocities = graph.y

    # Get velocity noise
    velocity_noise = get_velocity_noise(
        graph, noise_std=cfg.data.noise_std, device=device
    )
    with torch.no_grad():
        pred_acc, target_acc = simulator.predict_acceleration(
            current_velocities=current_velocities,
            node_type=node_types,
            edge_index=edge_index,
            edge_features=edge_features,
            target_velocities=target_velocities,
            velocity_noise=velocity_noise,
        )

        # Compute loss
        mask = torch.logical_or(
            node_types == NodeType.NORMAL, node_types == NodeType.OUTFLOW
        )
        errors = ((pred_acc - target_acc) ** 2)[
            mask
        ]  # only compute errors if node_types is NORMAL or OUTFLOW
        loss = torch.mean(errors)

    return loss


@hydra.main(version_base=None, config_path="..", config_name="config_mesh")
def main(cfg: Config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.hardware.cuda_device_number is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(cfg.hardware.cuda_device_number)}")

    # load simulator
    simulator = learned_simulator.MeshSimulator(
        simulation_dimensions=2,
        nnode_in=11,
        nedge_in=3,
        latent_dim=128,
        nmessage_passing_steps=15,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        nnode_types=3,
        node_type_embedding_size=cfg.data.node_type_embedding_size,
        device=device,
    )

    if cfg.mode == "train":
        train(simulator, cfg)
    elif cfg.mode in ["valid", "rollout"]:
        predict(simulator, device, cfg)


if __name__ == "__main__":
    main()
