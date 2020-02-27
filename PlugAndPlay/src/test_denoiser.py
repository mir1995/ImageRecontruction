import os
from input_data import datasetMRI
import torchvision
import torch
from DNN import Net
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
import numpy as np
import parameters


def loadModel(model_path):

    model = Net(parameters.Images.CHANNELS, parameters.Minimiser.NUMB_FEAT_MAPS)
    model = torch.nn.DataParallel(model) # don't know but has to do with the fact that DataParallel is used during training
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model


def plotGraph(path_to_data, save = None):

    import pandas as pd 

    fig, ax = plt.subplots(figsize=(8,6))


    df = pd.read_csv(path_to_data, index_col = None)

    # get root mean squared error instead
    # this is the average mean squared error

    for idx, gp in df.groupby(by = 'TrainedNoiselLevel'):
        gp.plot(x = 'TestNoiseLevel', y = 'NMSE', ax = ax, label=idx,  linestyle='--', marker='o')
    
    
    ax.set_ylabel('NMSE', labelpad = 15, fontsize = 16)
    ax.set_xlabel('Test Noise Level', labelpad = 15, fontsize = 16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_yscale('log')
    #plt.ylim(10**(-4),10**(-2))
    plt.xlim(-0.01, 0.175)
    plt.tight_layout()
    plt.legend(loc=2, prop={'size': 12}, title = 'CNN - Training Noise')
    if not save:
        plt.show()
        plt.close('all')
    else:
        plt.savefig(os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'plots', 'NMSE.pdf'))




def plotImages(model_path, imgs_path):

    # Load training dataset
    trainset = datasetMRI(imgs_path, parameters.Images.TRANSFORM)
    LOADER_TRAIN = torch.utils.data.DataLoader(trainset, batch_size=1)

    model = loadModel(model_path)

    model.eval()

    img_in = trainset[5]
    Tensor = torch.FloatTensor # for the plot I will always be running on locally
    imgs = [0]*2
    with torch.no_grad():
        
        sigma = 0.15
        img_noisy = img_in.unsqueeze(0)
        img_noisy = torch.autograd.Variable( 
                    img_noisy.type(Tensor), requires_grad=False)
        print(torch.mean(img_noisy))
        img_noisy = img_noisy + sigma*torch.randn(img_noisy.shape)
        img_noisy1 = img_noisy
        img_out = model(img_noisy)
        for i in range(2):
            imgs[i] = img_out.squeeze(0).squeeze(0).cpu().numpy().copy()
            img_noisy = img_out.clone()
            img_out = model(img_noisy)
        

    # plot images

    img_noisy1 = img_noisy1.squeeze(0).squeeze(0).cpu().numpy()

    img_in = img_in.squeeze(0)

    img_out = img_out.squeeze(0).squeeze(0).cpu().numpy()

    # print images
    for i in range(1):
        fig, axes = plt.subplots(1,3, figsize = (8,6))
        axes[0].imshow(imgs[0], cmap='gray')
        axes[1].imshow(img_in, cmap='gray')
        axes[2].imshow(imgs[1], cmap='gray')
        # labels = [r'Noisy $sigma = {}$'.format(sigma), 'Out', 'Truth']
        labels = ['One Pass', 'Truth', 'Two Passes']
        for i, ax in enumerate(axes):
            ax.set_xlim(64, 192)
            ax.set_ylim(64,192)
            ax.set_title(labels[i],pad = 20)
            ax.set_xticks([])
            ax.set_yticks([])
        ax2 = fig.add_subplot(338)
        ax2.margins(2, 2)           # Values >0.0 zoom out
        ax2.imshow(img_noisy1, cmap='gray')
        plt.xticks([]),plt.yticks([])
        plt.tight_layout()
        #plt.show()
        #plt.close()
    plt.savefig(os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'plots', 'CNN01Noise015TwoPasses.pdf'), bbox_inches='tight')
    #plt.close('all')


if __name__ == "__main__":

    plotImages(model_path = parameters.Models.DCNN_256_01 , imgs_path = parameters.Images.PATH_TEST)

    #############################################

    #plotGraph(parameters.Data.DENOISER_MSE, save = False) # should scale the sigma as well if you are reporting the noise # try simply adding back the noise 