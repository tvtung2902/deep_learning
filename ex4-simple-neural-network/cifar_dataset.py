from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import cv2

class CIFARDataset(Dataset):
    def __init__(self, root, train = True):
        if train:
            data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else :
            data_files = [os.path.join(root, "test_batch")]
        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, "rb") as f:
                data = pickle.load(f, encoding="bytes")
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index].reshape((3, 32, 32)).astype(np.float32)
        label = self.labels[index]
        return image / 255., label

if __name__ == '__main__':
    data_set = CIFARDataset('../all-data/data-cifar/cifar-10-batches-py', train = True)
    image, label = data_set[1]
    print(image)
    image = np.reshape(image, (3, 32, 32))
    print(image)
    image = np.transpose(image, (1, 2, 0))
    print(label)
    cv2.imshow('image', image)
    cv2.waitKey(0)
