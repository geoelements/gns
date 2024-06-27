import torch


def setup(rank, world_size, device):
    """Initializes distributed training.

    Args:
        rank (int): Rank of current process.
        world_size (int): Number of processes.
    """
    # Initialize group, blocks until all processes join.
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    """
    Clean up distributed training.
    """
    torch.distributed.destroy_process_group()


def spawn_train(train_fxn, flags, world_size, device):
    """Spawns distributed training.

    Args:
        train_fxn (function): Function to train model.
        flags (dict): Dictionary of flags.
        world_size (int): Number of processes.
        device (torch.device): torch device type
    """
    torch.multiprocessing.spawn(
        train_fxn, args=(flags, world_size, device), nprocs=world_size, join=True
    )
