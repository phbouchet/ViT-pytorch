import logging

import torch.distributed as dist

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, DistributedSampler

logger = logging.getLogger(__name__)

def get_loader(config):
    if config["local_rank"] not in [-1, 0]:
        dist.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((config["img_size"], config["img_size"]), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if config["dataset"] == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if config["local_rank"] in [-1, 0] else None

    else:
        raise NotImplementedError("No other dataset implemented. Please specify which dataset to use.")

    if config["local_rank"] == 0:
        dist.barrier()

    # Create sampler
    train_sampler = RandomSampler(trainset) if config["local_rank"] == -1 else DistributedSampler(trainset)
    test_sampler  = SequentialSampler(testset)


    train_loader = DataLoader(trainset,
                              sampler     = train_sampler,
                              batch_size  = config["train_batch_size"],
                              num_workers = config["num_workers"],
                              pin_memory  = True)

    test_loader = DataLoader(testset,
                             sampler     = test_sampler,
                             batch_size  = config["eval_batch_size"],
                             num_workers = config["num_workers"],
                             pin_memory  = True) if testset is not None else None

    return train_loader, test_loader
