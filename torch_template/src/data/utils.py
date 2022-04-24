from typing import Sized
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from random import seed as random_seed
from numpy.random import seed as numpy_seed

from torch import Generator as TorchGenerator
from torch import initial_seed as torch_initial_seed



def generate_dataloader(shuffle: bool, samples: Sized, batch_size: int):
    g = TorchGenerator()
    g.manual_seed(0)

    if shuffle:
        sampler = RandomSampler(samples, num_samples=len(samples), generator=g)
    else:
        sampler = SequentialSampler(samples)

    dataloader = DataLoader(
        samples,
        batch_size=batch_size,
        sampler=sampler,
        worker_init_fn=seed_worker,
        generator=g)

    return dataloader

def seed_worker(worker_id: int):
    worker_seed = torch_initial_seed() % 2**32
    numpy_seed(worker_seed)
    random_seed(worker_seed)
