import torch
<<<<<<< HEAD
import torch.distributed as dist
from torch.utils import collect_env
from torch.utils.data.distributed import DistributedSampler

from gns import data_loader
=======
>>>>>>> e3283cf4e5ff4e27c69fd6e9647e7a89c47df303


def setup(local_rank: int):
    """Initializes distributed training.
    """
    # Initialize group, blocks until all processes join.
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',
    )
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed(0)
    verbose = (dist.get_rank() == 0)

    if verbose:
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
    return verbose, world_size
      


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
<<<<<<< HEAD


def get_data_distributed_dataloader_by_samples(
    path, input_length_sequence, batch_size, shuffle=True
):
    """Returns a distributed dataloader.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle dataset.
    """
    dataset = data_loader.SamplesDataset(path, input_length_sequence)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=shuffle)
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=data_loader.collate_fn,
    )


=======
>>>>>>> e3283cf4e5ff4e27c69fd6e9647e7a89c47df303
