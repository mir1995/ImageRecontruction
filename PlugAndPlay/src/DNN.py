import torch


class Net(torch.nn.Module):
    def __init__(self, channels=1, features=32):
        super(Net, self).__init__()

        kernel_size = 3
        padding = 1

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3)
        # an affine operation: y = Wx + b
        # 6*6 from image dimension
        self.fc1 = torch.nn.Linear(1 * (features - 2)**2, features**2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = torch.nn.functional.relu(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":

    nrows = 5
    ncols = 5
    net = Net(1, nrows)
    input_ = torch.randn(1, 1, nrows, ncols, requires_grad=True)
    out = net(input_)
    print(out)
