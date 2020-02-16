import os
import torch
import torchvision


class Images:

    PATH_TRAINING = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data', 'training', '*.png')

    PATH_TEST = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data', 'test', '*.png')

    RESOLUTION = (125, 125)

    CHANNELS = 1

    TRANSFORM = torchvision.transforms.Compose(   # images to tensors
        [torchvision.transforms.ToTensor()])   # what does the normalisation do


class Minimiser:

    BATCH_SIZE = 9

    NUMB_FEAT_MAPS = 32

    NUMB_LAYERS = 10

    # Noise level
    SIGMA = 0.1

    # number of dataset iterations
    EPOCHS = 20

    # training setup
    CRITERION = torch.nn.MSELoss()

class Models:

    DCNN_125 = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'models', 'dncnn_125resolution.pth')


if __name__ == "__main__":

    print(Images.RESOLUTION)
