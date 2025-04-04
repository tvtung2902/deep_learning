from torchvision.datasets import  CIFAR10

train_dataset = CIFAR10(root='../all-data/data-cifar', train=True, download=True)
test_dataset = CIFAR10(root='../all-data/data-cifar', train=False, download=True)
index = 1000
image, label = train_dataset.__getitem__(index)
image.show()
print(image.size)