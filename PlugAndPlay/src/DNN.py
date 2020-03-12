import torch
import numpy as np
import myRelu
import GroupSort


class Net(torch.nn.Module):

    def __init__(self, channels=1, features=64, num_of_layers=10):
        super(Net, self).__init__()

        self.M = 12

        self.D = [64,64,16,16,4,4]

        #self.conv5 = torch.nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3, padding=1, bias=False)
        #layers.append(torch.nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        #layers.append(torch.nn.ReLU(inplace=True))
        #for _ in range(num_of_layers-2):
         #   layers.append(torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
          #  layers.append(torch.nn.BatchNorm2d(features))
           # layers.append(torch.nn.ReLU(inplace=True))
        #layers.append(torch.nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        #self.dncnn = torch.nn.Sequential(*layers)

    def forward(self, x):

        psi_singletons = [x]

        # the many outputs  
        psi_many = []

        # the single frames output of the previous ...
        phi_singletons = []

        phi_many = []

        count = 0

        for d in self.D:

            output = torch.nn.Conv2d(in_channels = 1, out_channels = d, kernel_size = np.sqrt(d) + 1, padding=1, bias=False)(psi_singletons[count])

            many, one = torch.split(output, d - 1, dim=0) 

            psi_singletons.append(one)

            psi_many.append(many) 

            count += 1
        
        phi_singletons.append(psi_singletons[-1])

        # transform the many output of the convolution
        phi_many = list(map(self.nonLinear, psi_many))   ## only this one remaining

        for i in range(self.M/2):
            

            # join the phi many with phi singletons
            phi = torch.cat([phi_many[self.M/2 - 1 - i],phi_singletons[i]],1)

            # apply many to one convolution onto last psi and the transformed psi
            phi_singletons.append(torch.nn.Conv2d(in_channels = self.D[self.M/2 - 1 - i], out_channels = 1, kernel_size = np.sqrt(self.D[self.M/2 - 1 - i]) + 1, padding=1, bias=False)(phi))


        return phi_singletons[-1]



    def nonLinear(self, x):

        x1 = x.clone()

        y1 = myRelu.relu(torch.rand(1))(x1)

        y2 = - myRelu.relu(torch.rand(1))(x1)

        z = GroupSort.groupSort(_max_ = True, torch.rand(1), torch.rand(1))(y1, y2)

        return GroupSort.groupSort(_max_ = False, torch.rand(1), torch.rand(1))(x, z)









        





            






        x1 = self.dncnn(x)
        #print(x1.size)
        out = (x+x1)/2    # skip connection + /2 (non-expansiveness, firmly?)
        return out

if __name__ == "__main__":

    nrows = 5
    ncols = 5
    net = Net(1, nrows)
    input_ = torch.randn(1, 1, nrows, ncols, requires_grad=True)
    out = net(input_)
    print(out)
