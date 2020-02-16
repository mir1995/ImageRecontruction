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


# Load training dataset
trainset = datasetMRI(parameters.Images.PATH_TRAINING, parameters.Images.TRANSFORM)
LOADER_TRAIN = torch.utils.data.DataLoader(trainset, batch_size=1)


model = Net(parameters.Images.CHANNELS, parameters.Minimiser.NUMB_FEAT_MAPS)
model = torch.nn.DataParallel(model) # don't know but has to do with the fact that DataParallel is used during training
model.load_state_dict(torch.load(parameters.Models.DCNN_125, map_location=torch.device('cpu')))
model.eval()

img_in = trainset[0]
Tensor = torch.FloatTensor
imgs = [0]*7
with torch.no_grad():
    
    sigma = 0.2
    img_noisy = img_in.unsqueeze(0)
    img_noisy = torch.autograd.Variable( 
                img_noisy.type(Tensor), requires_grad=False)
    img_noisy = img_noisy + sigma*torch.randn(img_noisy.shape)
    img_noisy1 = img_noisy
    img_out = model(img_noisy)
    for i in range(7):
        imgs[i] = img_out.squeeze(0).squeeze(0).cpu().numpy().copy()
        img_noisy = img_out.clone()
        img_out = model(img_noisy)
    

# plot images

img_noisy1 = img_noisy1.squeeze(0).squeeze(0).cpu().numpy()

img_in = img_in.squeeze(0)

img_out = img_out.squeeze(0).squeeze(0).cpu().numpy()

# print images
for i in range(7):
    fig, axes = plt.subplots(1,3, figsize = (10,8))
    axes[0].imshow(img_noisy1, cmap='gray')
    axes[1].imshow(imgs[i], cmap='gray')
    axes[2].imshow(img_in, cmap='gray')
    labels = [r'Noisy $sigma = {}$'.format(sigma), 'Out', 'Truth']
    for i, ax in enumerate(axes):
        ax.set_title(labels[i],pad = 20)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.xticks([]),plt.yticks([])
    plt.show()
    plt.close()
#plt.savefig(os.path.join(os.path.sep, os.path.dirname(
        #os.path.dirname(os.path.abspath(__file__))), 'plots', 'noisy2OutTruth.pdf'))
plt.close('all')