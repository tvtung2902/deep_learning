from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms.v2 import ToTensor, Resize
from animal_dataset import AnimalDataset

if __name__ == "__main__":
    # training_data = CIFAR10(root='../all-data/data-cifar', train=True, transform = ToTensor())
    # image, label = training_data.__getitem__(123)
    # image.show()
    # drop_last: remove final batch if batch size is not enough
    # totensor: convert a PIL image or ndarray to tensor (C x H x W) (0 -> 1)
    transform = Compose([
        Resize((200, 200)),
        ToTensor(),
    ])

    training_data = AnimalDataset(root='../all-data/animals', train=True, transform=transform)
    train_loader = DataLoader(training_data,
                              batch_size=16,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True,
                              )

    for images, labels in train_loader:
        print(images.shape)
        print(labels)