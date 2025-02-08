import logging

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from utils.hymenoptera_dataset import BeeAntDataset

logger = logging.getLogger(__name__)


def get_loader(config):
    if config.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.RandomResizedCrop((config.img_size, config.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = BeeAntDataset(root_dir="./data/hymenoptera_data/train",
                             transform=transform_train)

    testset  = BeeAntDataset(root_dir="./data/hymenoptera_data/val",
                             transform=transform_test)

    if config.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if config.local_rank == -1 else DistributedSampler(trainset)
    test_sampler  = SequentialSampler(testset)

    train_loader = DataLoader(trainset,
                              sampler     = train_sampler,
                              batch_size  = config.train_batch_size,
                              num_workers = 4,
                              pin_memory  = True)
    test_loader = DataLoader(testset,
                             sampler      = test_sampler,
                             batch_size   = config.eval_batch_size,
                             num_workers  = 4,
                             pin_memory   = True) if testset is not None else None

    return train_loader, test_loader
