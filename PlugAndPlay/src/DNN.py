import torch


class Net(torch.nn.Module):

    def __init__(self, channels=1, features=64, num_of_layers=10):
        super(Net, self).__init__()

        kernel_size = 3
        padding = 1 # ensures input feature map = output feature map

        layers = []
        layers.append(torch.nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(torch.nn.BatchNorm2d(features))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = torch.nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.dncnn(x)
        #print(x1.size)
        out = x+x1
        return out

if __name__ == "__main__":

    nrows = 5
    ncols = 5
    net = Net(1, nrows)
    input_ = torch.randn(1, 1, nrows, ncols, requires_grad=True)
    out = net(input_)
    print(out)
