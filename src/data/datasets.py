import numpy as np
import torch
import torchvision
import torchvision.transforms as T


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
