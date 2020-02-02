import torch


class Net(torch.nn.Module):
    def __init__(self, channels=1, features=32, num_of_layers=17):
        super(Net, self).__init__()

        kernel_size = 3
        padding = 1

        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)         # how do I know what is the size of the output image?

    def forward(self, x):

        return x + self.dncnn(x) # skip connection? Enquire about the advantage w.r.t. learning directly the doising

net = Net()