from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


def get_shape(data_loader):
    images, labels = next(iter(data_loader))
    return images.shape, labels.shape


def fashion_loader(dir_path='./data', transform=None, batch_size=64, shuffle=True, drop_last=True, num_workers=0):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),])

    trainset = FashionMNIST(root=dir_path, train=True, download=True, transform=transform,)
    testset = FashionMNIST(root=dir_path, train=False, download=True, transform=transform,)

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def cifar_loader(dir_path='./data', transform=None, batch_size=64, shuffle=True, drop_last=True, num_workers=0):
    if transform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    trainset = CIFAR10(root=dir_path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=dir_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_loader, test_loader