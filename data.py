import torch
from torchvision import datasets, transforms
import ssl

def get_cifar10():
    ssl._create_default_https_context = ssl._create_unverified_context

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

if __name__ == "__main__":

    train_dataset, test_dataset = get_cifar10()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)

    train_loader = iter(train_loader)
    images, _ = next(train_loader)
    print(images.max(), images.min())
    print(images.shape)