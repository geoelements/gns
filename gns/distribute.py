import torch
from torch.utils.data.distributed import DistributedSampler

from gns import data_loader

def setup(rank, world_size):
    # Initialize group, blocks until all processes join.
    torch.distributed.init_process_group(backend="nccl",
                                         rank=rank,
                                         world_size=world_size,
                                        )

def cleanup():
    torch.distributed.destroy_process_group()


def spawn_train(train_fxn, flags, world_size):
    torch.multiprocessing.spawn(train_fxn,
                                args=(flags, world_size),
                                nprocs=world_size,
                                join=True
                               )


def get_data_distributed_dataloader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    dataset = data_loader.SamplesDataset(path, input_length_sequence)
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    return torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size,
                                       pin_memory=True, collate_fn=data_loader.collate_fn)