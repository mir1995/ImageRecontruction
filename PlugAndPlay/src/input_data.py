import os
import glob
import numpy as np
from PIL import Image
import torch
import torchvision


class datasetMRI(torch.utils.data.Dataset):

    def __init__(self, path_data, transf=None, resolution=None):

        # plus if you want to add the ones from the test data - but rather keep them separate
        self.filenames = glob.glob(path_data)
        self.transform = transf
        self.resolution = resolution
        self.len = len(self.filenames)

    def __getitem__(self, index):
        """ 
        Return a sample img from the training 
        dataset of type 'numpy array'.
        """

        # opens and identifies the img but data not read until process or load()? / opens at the path
        img_true = Image.open(self.filenames[index])

        if self.resolution:
            img_true.resize(self.resolution, Image.ANTIALIAS)

        if self.transform:
            img_true = self.transform(img_true)

        return img_true  # return image as an array

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import parameters

    sample_img = datasetMRI(parameters.Images.PATH_TRAINING)[20] 
    plt.imshow(sample_img, cmap='gray')
    plt.show()
    plt.close('all')
