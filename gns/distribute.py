import torch
from torch.utils.data.distributed import DistributedSampler

from gns import data_loader

def setup(rank, world_size, device):
    """Initializes distributed training.
    
    Args:
        rank (int): Rank of current process.
        world_size (int): Number of processes.
    """
    # Initialize group, blocks until all processes join.
    torch.distributed.init_process_group(backend="nccl",
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
    torch.multiprocessing.spawn(train_fxn,
                                args=(flags, world_size, device),
                                nprocs=world_size,
                                join=True
                                )


def get_data_distributed_dataloader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    """Returns a distributed dataloader.
    
    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle dataset.
    """
    dataset = data_loader.SamplesDataset(path, input_length_sequence)
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    return torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size,
                                       pin_memory=True, collate_fn=data_loader.collate_fn)