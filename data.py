import torch
import torchvision
import ssl

def get_cifar10():
    ssl._create_default_https_context = ssl._create_unverified_context
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = get_cifar10()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)