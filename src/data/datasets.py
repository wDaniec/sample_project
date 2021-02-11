import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data.sampler import SubsetRandomSampler

def cifar(variant="10", seed=777, batch_size=128, num_workers=8):
    train_dataset, valid_dataset = get_cifar_dataset(variant, seed, batch_size, num_workers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    return train_loader, valid_loader

def half_cifar(variant="10", seed=777, batch_size=128, num_workers=8):
    train_dataset, valid_dataset = get_cifar_dataset(variant, seed, batch_size, num_workers)
    # print(train_dataset, valid_dataset)
    train_length = len(train_dataset)
    half_length = int(0.5 * train_length)
    subset_indices = torch.randperm(train_length)[:half_length]

    # # napraw ten brzydki kod tutaj
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, sampler=SubsetRandomSampler(subset_indices)
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size,  num_workers=num_workers, shuffle=True
    )
    return train_loader, valid_loader


def get_cifar_dataset(variant="10", seed=777, batch_size=128, num_workers=8):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std),
    ])

    if variant == "10":
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar100', train=True, download=True, transform=transform)
        valid_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar100', train=False, download=True, transform=transform)
        assert len(train_dataset) == 50000
    elif variant == "100":
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data/cifar100', train=True, download=True, transform=transform)
        valid_dataset = torchvision.datasets.CIFAR100(
            root='./data/cifar100', train=False, download=True, transform=transform)
        assert len(train_dataset) == 50000
    else:
        raise NotImplementedError()

    return train_dataset, valid_dataset


def fmnist(seed=777, batch_size=128, num_workers=8):
    torch.manual_seed(seed)
    fmnist_mean = (0.5,)
    fmnist_std = (0.5,)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(fmnist_mean, fmnist_std),
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST', train=True, download=True, transform=transform)
    valid_dataset = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    return train_loader, valid_loader
