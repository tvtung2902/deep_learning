import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    # define các layer
    def __init__(self, num_classes=10):
        super().__init__()
        # làm phẳng ngoại trừ batch size
        self.flatten = nn.Flatten()
        # in_features: size phụ thuộc vào input, out_features: k có rule nhưng số lượng feature thường là lũy thừa của 2
        # sẽ tăng rồi sau đó giảm dần đến khi = output feature = số lượng class
        # self.fc1 = nn.Linear(in_features=32*32*3, out_features=256)
        # self.act1 = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(in_features= 32 * 32 * 3, out_features= 256),
            nn.ReLU(),
            nn.Linear(in_features= 256, out_features= 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.ReLU(),
        )

    # define các kết nối giữa các layer
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = SimpleNeuralNetwork()
    input_data = torch.randn(8, 3, 32, 32)
    result = model(input_data)
    print(result)