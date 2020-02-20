import torch
import parameters
from input_data import datasetMRI
from DNN import Net
from trainDNN import checkGPU


def getMSE(loader_test, net, sigma, criterion, numb_itrs = 10):

    # check availability GPU and return appropriate Tensor module?
    net, Tensor = checkGPU(net)

    mean_loss = 0
    var = 0

    for _ in range(numb_itrs):

        with torch.no_grad():

            for data in loader_test:
                net.eval()  # not sure
                # zero gradients otherwise it accumulates them?
                data_true = torch.autograd.Variable(  # does this turn the image in the same dimension of the network output??
                    data.type(Tensor), requires_grad=False)  # Keep initial data in memory ## ?? should not this be set to true or is it false when dealing with pretrained nets? or more simply not a parameter??
                noise = sigma * torch.randn(data_true.shape).type(Tensor)
                # Create noisy data
                data_noisy = data_true + noise

                # forward + loss
                out = net(data_noisy)
                loss = criterion(out*255, data_true*255)

                # do not know what this does
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)

                mean_loss += loss.item()
                var += loss.item()**2

    # average loss
    mean_loss /= (len(loader_test) * numb_itrs)
    var/=  (len(loader_test) * numb_itrs) 
    print(len(loader_test))
    var-=  (mean_loss)**2

    return mean_loss, var



if __name__ == "__main__":
    
    
    import os
    import csv

    STATS_FILE = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))),'data', 'stats', 'mse.csv')

    loader_test = torch.utils.data.DataLoader(
        datasetMRI(parameters.Images.PATH_TEST,
                   transf=parameters.Images.TRANSFORM),
        parameters.Minimiser.BATCH_SIZE)
    print(loader_test)

    net = Net(parameters.Images.CHANNELS, parameters.Minimiser.NUMB_FEAT_MAPS)
    # don't know but has to do with the fact that DataParallel is used during training
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(
        parameters.Models.DCNN_256_005, map_location=torch.device('cpu')))

    with open(STATS_FILE, 'a') as csv_file:

        stats_writer = csv.writer(csv_file, delimiter=',')

        #stats_writer.writerow(['TrainedNoiselLevel', 'TestNoiseLevel', 'MSE', 'VAR'])

        sigma = 0.05
        for sigma_test in [0.05, 0.08, 0.1, 0.15, 0.2]:
            mean, var = getMSE(loader_test, net, sigma_test, torch.nn.MSELoss())
            stats_writer.writerow([sigma, sigma_test, mean, var ])


