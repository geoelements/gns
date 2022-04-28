import torch
from torch.utils.data.distributed import DistributedSampler

import data_loader

def setup(rank, world_size):
    # Initialize group, blocks until all processes join.
    torch.distributed.init_process_group("gloo",
                                         rank=rank,
                                         world_size=world_size,
                                         backend="nccl"
                                        )

def cleanup():
    torch.distributed.destroy_process_group()


def spawn_train(train_fxn, world_size):
    torch.multiprocessing(train_fxn,
                          args=(world_size,),
                          nproces=world_size,
                          join=True)

def get_data_distributed_dataloader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    dataset = data_loader.SamplesDataset(path, input_length_sequence)
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    return torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)