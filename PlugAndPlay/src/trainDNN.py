import torch
import torchvision
import os
from DNN import Net
from input_data import datasetMRI

"""
nrows = 5
ncols = 5
net = Net(1, nrows)

.requires_grad as True, it starts to track all operations on it. 
When you finish your computation you can call .backward() and have all the gradients computed automatically.
The gradient for this tensor will be accumulated into .grad attribute.
If you want to compute the derivatives, you can call .backward() on a Tensor.
If Tensor is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to backward(),
however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape.

input_ = torch.randn(1, 1, nrows, ncols,
                     requires_grad=True)  # automatic differentation
# with torch.no_grad():
out = net(input_)
input_ = input_.view(1, -1)
print(input_.size())
out.backward(input_)
print(input_.grad)
"""


def checkGPU(net):
    # Move to GPU if possible
    cuda = True if torch.cuda.is_available() else False

    if cuda:
    
        print("cuda driver found - using a GPU.\n")
        net.cuda()  # ?

        return torch.nn.DataParallel(net).cuda(), torch.cuda.FloatTensor  # not sure what this does

    else:

        print("no cuda driver found - using a CPU.\n")

        return torch.nn.DataParallel(net), torch.FloatTensor  # can stil parallelise on CPU?

def createCheckpoint():
    # ------------------------------
    # Create a checkpoint folder to save network during training
    # ------------------------------
    checkpoints_folder = '../checkpoints/'
    try:  # Create checkpoint directory
        os.mkdir(checkpoints_folder)
    except:
        print('folder '+checkpoints_folder+' exists')

    return None


def main(loader_train, net, sigma, epochs, criterion, optimizer):
    """
    First check for GPU and then train network
    """

    net, Tensor = checkGPU(net)

    createCheckpoint()

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(loader_train, 0):

            optimizer.zero_grad() # otherwise it accumulates the gradiensts?

            # ------------------------------
            # Create noisy data
            # ------------------------------
            data_true = torch.autograd.Variable(
                data.type(Tensor), requires_grad=False)  # Keep initial data in memory ## why not true
            # what is the advantage of noising each image every time? in some sense you are changhing the dataset each time - for underfitting?
            noise = sigma * torch.randn(data_true.shape).type(Tensor) # requires_grad should be true here then?? automatics?
            data_noisy = data_true+noise

            # forward + backward + optimize
            out = net(data_noisy)
            loss = criterion(outputs, data_true)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # do not know what this does

            optimizer.step()

            print("[epoch %d][%d/%d] loss: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item()), end='\r')   # why isit not printing this one out?

            loss_tot += loss.item()

        loss_tot /= len(loader_train)

        torch.save(model.state_dict(), os.path.join(
            checkpoints_folder, 'dncnn_toy_'+str(epoch)+'.pth'))

        print("[epoch %d]: average training loss: %.4f" %
              (epoch+1, loss_tot))

    print('Finished Training')

# WRIRTE THE ABOCE AS A FUNCTION


if __name__ == "__main__":

    # set up parameters

    # ------------------------------
    # Load training dataset
    # ------------------------------
    from input_data import datasetMRI

    PATH_IMG = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data', 'trainingset', '*.png')

    TRANSFORM = torchvision.transforms.Compose(   # images to tensors
        [torchvision.transforms.ToTensor()])   # what does the normalisation do - do not know yet - when is it needed

    BATCH_SIZE = 9

    trainset = datasetMRI(PATH_IMG, TRANSFORM)

    LOADER_TRAIN= torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffles the data set at each epoch

    NET = Net(1, 256)

    # Noise level
    SIGMA = 0.1
    # number of dataset iterations
    EPOCHS = 2

    # training setup

    CRITERION = torch.nn.MSELoss()
    # https://arxiv.org/pdf/1412.6980.pdf why have to pass in net.parameters
    OPTIMIZER = torch.optim.Adam(NET.parameters(), lr=1e-3)

    main(loader_train = LOADER_TRAIN, net=NET, sigma=SIGMA, epochs=EPOCHS,
         criterion=CRITERION, optimizer=OPTIMIZER)
