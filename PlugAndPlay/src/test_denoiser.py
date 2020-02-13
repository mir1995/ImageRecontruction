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
model.load_state_dict(torch.load(os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'models', 'dncnn_toy_19.pth'), map_location=torch.device('cpu')))
model.eval()

img_in = trainset[0]
Tensor = torch.FloatTensor

with torch.no_grad():
    
    sigma = 0.1
    img_noisy = img_in.unsqueeze(0)
    img_noisy = torch.autograd.Variable( 
                img_noisy.type(Tensor), requires_grad=False)
    img_noisy = img_noisy + sigma*torch.randn(img_noisy.shape)
    img_out = model(img_noisy)
    

# plot images
fig, ax = plt.subplots(1,3, figsize = (10,8))

img_noisy = img_noisy.squeeze(0).squeeze(0).cpu().numpy()

img_out = img_out.squeeze(0).squeeze(0).cpu().numpy()

# print images
ax[0].imshow(img_noisy, cmap='gray')
ax[1].imshow(img_out, cmap='gray')
ax[2].imshow(img_in.squeeze(0), cmap='gray')
plt.show()
plt.close('all')