import torch
import numpy as np
import myRelu
import GroupSort
import parameters


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.M = 12

        self.D = [64, 64, 16, 16, 4, 4]

        self.linears = torch.nn.ModuleList([])

        for d in self.D:

            pad = int(np.sqrt(d))//2

            self.linears.append(torch.nn.Conv2d(in_channels=1, out_channels=d, kernel_size=int(
                np.sqrt(d)) + 1, padding=pad, bias=False))

        for _ in self.D:

            self.linears.extend([myRelu.relu(torch.rand(1)),
                            myRelu.relu(torch.rand(1)),
                            GroupSort.groupSort(True, torch.rand(1), torch.rand(1)),
                            GroupSort.groupSort(False, torch.rand(1), torch.rand(1))])

        for i in range(self.M//2):

            pad=int(np.sqrt(self.D[self.M//2 - 1 - i]))//2

            self.linears.append(torch.nn.Conv2d(in_channels=self.D[self.M//2 - 1 - i], out_channels=1, kernel_size=int(
                np.sqrt(self.D[self.M//2 - 1 - i])) + 1, padding=pad, bias=False))



    def forward(self, x):

        ID=x.clone()

        psi_singletons=[x]

        # the many outputs
        psi_many=[]

        # the single frames output of the previous ...
        phi_singletons=[]

        phi_many=[]

        global ix 
        
        ix=0

        for d in self.D:

            # operation
            output = self.linears[ix](psi_singletons[ix])

            
            many, one = torch.split(output, d - 1, dim=1)
            
            psi_singletons.append(one)

            psi_many.append(many)

            ix += 1

        phi_singletons.append(psi_singletons[-1])

        # operation - do loop instead
        for i in range(self.M//2):
            # transform the many output of the convolution
            phi_many.append(self.nonLinear(psi_many[i]))  # only this one remaining
            ix += 1

        for i in range(self.M//2):


            # join the phi many with phi singletons
            phi=torch.cat([phi_many[self.M//2 - 1 - i], phi_singletons[i]], 1)

            # operations
            # apply many to one convolution onto last psi and the transformed psi
            phi_singletons.append(self.linears[ix](phi))

            ix += 1


        return (phi_singletons[-1] + ID)/2



    def nonLinear(self, x):


        global ix

        x1=x.clone()

        y1=self.linears[ix](x1)   # must write as tensor module
        
        ix += 1

        y2= - self.linears[ix](x1)

        ix += 1

        z= self.linears[ix](y1, y2)

        ix += 1

        return self.linears[ix](x, z)



if __name__ == "__main__":

    net=Net()
    for name, param in net.state_dict().items():
        print(name, param)

    print(net.linears[0])
    net.linears[0].weight = torch.nn.Parameter(torch.tensor(0) * torch.rand(size = net.linears[0].weight.size()))
    print(net.linears[0].weight)
    #param = list(net.parameters())[0]
    #print(param)
    #print(list(net.parameters())[-6:])
    #print(list(net.parameters())[30].size())
    #print(len(list(net.parameters())[-6:]))

    # print(net)
    input_=torch.randn(1, 1, 256, 256, requires_grad=True)
    out=net(input_)
    #print(out)
