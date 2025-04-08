import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from animal_dataset import AnimalDataset
from sklearn.metrics import classification_report, accuracy_score
from model import SimpleCNN
from tqdm import tqdm
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="CNN training")

    parser.add_argument("--root", "-n", type=str, default='../all-data/animals', help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="Logging")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    transform = Compose(
        [
            Resize((args.image_size, args.image_size)),
            ToTensor(),
        ]
    )

    train_dataset = AnimalDataset(root=args.root, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    test_dataset = AnimalDataset(root=args.root, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    model = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            #forward
            output = model(images)
            loss = criterion(output, labels)
            progress_bar.set_description("epoch {}/{} iteration {}/{} loss {:.3f}".format(epoch + 1, num_epochs, i + 1, len(train_loader), loss))

            #backward
            optimizer.zero_grad() # refresh buffer
            loss.backward() # gradient
            optimizer.step()  # update parameter

        model.eval()
        all_predictions = []
        all_labels = []
        for i, (images, labels) in enumerate(test_loader):
            all_labels.extend(labels)
            # khong tinh gradient trong with torch.no_grad():
            with torch.no_grad():
                predictions = model(images) # 16 * 10
                indices = torch.argmax(predictions,1)
                all_predictions.extend(indices)
                loss = criterion(predictions, labels)

        all_predictions = [prediction.item() for prediction in all_predictions]
        all_labels = [label.item() for label in all_labels]

        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print(classification_report(all_labels, all_predictions))
