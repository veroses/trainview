from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader
