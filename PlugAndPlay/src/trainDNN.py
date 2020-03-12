
""" Extra notes
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
import os
import torch
import torchvision
from DNN import Net
from input_data import datasetMRI
import parameters
import numpy as np


def checkGPU(net):
    """
        Check GPU availability. 
        Nonetheless return ... (not sure what they do exactly)
    """
    cuda = True if torch.cuda.is_available() else False

    if cuda:

        print("cuda driver found - using a GPU.\n")
        net.cuda()  # ? see Audrey's noteboook

        # not sure what this second part does
        return torch.nn.DataParallel(net).cuda(), torch.cuda.FloatTensor

    else:

        print("no cuda driver found - using a CPU.\n")

        # can stil parallelise on CPU?
        return torch.nn.DataParallel(net), torch.FloatTensor


def createCheckpoint():
    """
        Creata a folder to save the model at the end of each epoch.
    """
    checkpoints_folder = '../models/'
    try:  # Create checkpoint directory
        os.mkdir(checkpoints_folder)
    except:
        print('folder '+checkpoints_folder+' exists')

    return checkpoints_folder


def main(loader_train, net, sigma, epochs, criterion, optimizer_kernels, optimizer_activation):
    """
    Train the network and save model after each epoch.
    Add some statistics (validation) to keep track of performance.
    """
    
    # check availability GPU and return appropriate Tensor module?
    net, Tensor = checkGPU(net)
    # create folder to save model
    checkpoints_folder = createCheckpoint()
    print(len(loader_train))
    # loop over the dataset multiple times
    for epoch in range(epochs):
        # Audrey had this why?
        # net.train()
        # compute average loss at each epoch
        loss_tot = 0.0

        # loop trough each batch?
        for i, data in enumerate(loader_train, 0):
            net.train()  # not sure
            # zero gradients otherwise it accumulates them?
            optimizer.zero_grad()
            # not sure of the following
            data_true = torch.autograd.Variable(  # does this turn the image in the same dimension of the network output??
                data.type(Tensor), requires_grad=False)  # Keep initial data in memory ## ?? should not this be set to true or is it false when dealing with pretrained models? or more simply not a parameter??
            s = float(np.random.choice(sigma))
            n = parameters.Images.RESOLUTION[0]
            complex_noise = np.random.normal(size=(n, n, 2)).view(
                np.complex128).reshape((n, n))
            noise = s * torch.from_numpy((np.fft.ifftshift(np.fft.fftshift(
                np.fft.fft2(complex_noise)))/n).reshape((n, n))).type(Tensor)
            #noise = s * torch.randn(data_true.shape).type(Tensor)
            # Create noisy data
            data_noisy = data_true + noise

            # forward + backward + optimize
            out = net(data_noisy)
            loss = criterion(out, data_true)
            loss.backward()   # computes gradients w.r.t. everything but does not update

            # do not know what this does
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # update kernel weights only
            optimizer_kernels.step()

            print("[epoch %d][%d/%d] loss: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item()), end='\r')   # why isit not printing this one out?

            loss_tot += loss.item()

        # average loss
        loss_tot /= len(loader_train)

        # save model

        print("[epoch %d]: average training loss: %.4f" %
              (epoch+1, loss_tot))
    torch.save(net.state_dict(), parameters.TRAINING_MODEL.model)

    print('Finished Training')


if __name__ == "__main__":

    # Load training dataset
    trainset = datasetMRI(parameters.Images.PATH_TRAINING,
                          parameters.Images.TRANSFORM,
                          parameters.Images.RESOLUTION)
    LOADER_TRAIN = torch.utils.data.DataLoader(
        trainset, batch_size=parameters.Minimiser.BATCH_SIZE)  # shuffles the data set at each epoch

    ##################### PARAMETERS #####################
    # initialise network
    NET = Net(parameters.Images.CHANNELS,
              parameters.Minimiser.NUMB_FEAT_MAPS,
              parameters.Minimiser.NUMB_LAYERS)

    main(loader_train=LOADER_TRAIN,
         net=NET,
         sigma=parameters.Minimiser.SIGMA,
         epochs=parameters.Minimiser.EPOCHS,
         criterion=parameters.Minimiser.CRITERION,
         optimizer_kernels=torch.optim.SGD(parameters.Parameters.KERNELS),
         optimizer_activation = torch.optim.SGD(parameters.Parameters.CONVOLUTIONS))  # https://arxiv.org/pdf/1412.6980.pdf why have to pass in net.parameters
