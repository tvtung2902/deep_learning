import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from cifar_dataset import CIFARDataset
from model_1 import SimpleNeuralNetwork
from sklearn.metrics import classification_report

if __name__ == '__main__':
    num_epochs = 100

    train_dataset = CIFARDataset(root='../all-data/data-cifar/cifar-10-batches-py', train=True)
    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True,
                                  )

    test_dataset = CIFARDataset(root='../all-data/data-cifar/cifar-10-batches-py', train=False)
    test_dataloader = DataLoader(dataset= test_dataset,
                                 batch_size=16,
                                 shuffle=False,
                                 num_workers=4,
                                 drop_last=False,
                                 )

    model = SimpleNeuralNetwork(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            #forward
            output = model(images)
            loss = criterion(output, labels)
            # print("epoch {}/{} iteration {}/{} loss {}".format(epoch + 1, num_epochs, i + 1, len(train_dataloader), loss))

            #backward
            optimizer.zero_grad() # refresh buffer
            loss.backward() # gradient
            optimizer.step()  # update parameter

        model.eval()
        all_predictions = []
        all_labels = []
        for i, (images, labels) in enumerate(test_dataloader):
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