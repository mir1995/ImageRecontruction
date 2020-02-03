import os
import glob

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

import torch
import torchvision


class datasetMRI(torch.utils.data.Dataset):

    def __init__(self, path_data, transf=None):

        # plus if you want to add the ones from the test data - but rather keep them separate
        self.filenames = glob.glob(path_data)
        self.transform = transf
        self.len = len(self.filenames)

    def __getitem__(self, index):
        """ 
        Return a sample img from the training 
        dataset of type 'numpy array'.
        """

        # opens and identifies the img but data not read until process or load()? / opens at the path
        img_true = Image.open(self.filenames[index])
        img_true = np.asarray(img_true)  # gray scale image

        if self.transform:
            img_true = self.transform(img_true)
            # print(img_true.size())

        return img_true  # return image as an array
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

if __name__ == "__main__":

    # think about where to have this - perhaps in trainDNN.py
    PATH_IMG = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data', 'trainingset', '*.png')

    TRANSFORM = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])   # what does the normalisation do - do not know yet - when is it needed

    BATCH_SIZE = 9

    # image as tensor object  # i am not doing any validation thing at the moment
    trainset = datasetMRI(PATH_IMG, TRANSFORM)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True)  # shuffles the data set at each epoch

    sample_img = datasetMRI(PATH_IMG)[10]
    # plot an image grayscale
    plt.imshow(sample_img, cmap='gray')
    plt.show()
    plt.close('all')

    # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

