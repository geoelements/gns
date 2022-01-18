import torch

from train import prepare_input_data

class TFIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self):
        super(TFIterableDataset).__init__():
        self.ds = prepare_input_data(FLAG.data_path,
                                     batch_size=FLAGS.batch_size)

    def __iter__(self):
        return self.ds


def get_data_loader(**kwargs):
    ds = TFIterableDataset()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=args.backend.size(),
        rank=args.backend.rank()
    )

    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size * args.batches_per_allreduce,
        sampler=train_sampler,
        **kwargs
    )

    return train_loader
