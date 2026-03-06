import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def get_data_loaders(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root=path, train=True,
                                          download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testSet = torchvision.datasets.MNIST(root=path, train=False,
                                         download=True, transform=transform)

    testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=False)

    return trainLoader, testLoader