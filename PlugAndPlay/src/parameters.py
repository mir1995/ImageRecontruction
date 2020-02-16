import os
import torch
import torchvision


class Images:

    PATH_TRAINING = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data', 'training', '*.png')

    PATH_TEST = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data', 'test', '*.png')

    # when None it will train on the original resolution, ow pass in tuple (,) for desired resolution
    RESOLUTION = None

    CHANNELS = 1

    TRANSFORM = torchvision.transforms.Compose([
        # torchvision.transforms.RandomResizedCrop(64), # still not clear # i think it could be a disadvantage as much as an advantage especially for classification - obviously depending on the image
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor()
    ])   # what does the normalisation do
    # torchvision.transforms.FiveCrop(size) # Crop the given PIL Image into four corners and the central crop


class Minimiser:

    BATCH_SIZE = 9

    NUMB_FEAT_MAPS = 32

    NUMB_LAYERS = 10

    # Noise level
    SIGMA = [0.1, 0.15, 0.2, 0.3]

    # number of dataset iterations
    EPOCHS = 20

    # training setup
    CRITERION = torch.nn.MSELoss()


class Models:

    DCNN_125_01 = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'models', 'dncnn_125resolution.pth')

    DCNN_256_01 = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'models', 'dncnn_256resolution_01noise.pth')
    
    DCNN_256_015 = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'models', 'dncnn_256resolution_015noise.pth')

    DCNN_256_02 = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'models', 'dncnn_256resolution_02noise.pth')

    DCNN_256_010150203 = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'models', 'dncnn_256resolution_varnoise.pth')

    DCNN_256_03 = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'models', 'dncnn_256resolution_03noise.pth')

if __name__ == "__main__":

    print(Images.RESOLUTION)
