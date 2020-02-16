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

PATH_IMG = os.path.join(os.path.sep, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'data', 'test', '*.png')
TRANSFORM = torchvision.transforms.Compose(   # images to tensors
        [torchvision.transforms.ToTensor()])   # what does the normalisation do - do not know yet - when is it needed

# Load training dataset
trainset = datasetMRI(PATH_IMG, TRANSFORM)
LOADER_TRAIN = torch.utils.data.DataLoader(trainset, batch_size=1)


model = Net(1, 32)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(parameters.Models.DCNN_125, map_location=torch.device('cpu')))
model.eval()

img_in = trainset[0]
Tensor = torch.FloatTensor

with torch.no_grad():
    
    sigma = 0.1
    img_noisy = img_in.unsqueeze(0)
    img_noisy = torch.autograd.Variable( # does this turn the image in the same dimension of the network output??
                img_noisy.type(Tensor), requires_grad=False)
    img_noisy = img_noisy + sigma*torch.randn(img_noisy.shape)
    img_out = model(img_noisy)
    

# plot images
fig, ax = plt.subplots(1,3, figsize = (10,8))

img_noisy = img_noisy.squeeze(0).squeeze(0).cpu().numpy()
#img_noisy = np.moveaxis(img_noisy, 0, -1)

img_out = img_out.squeeze(0).squeeze(0).cpu().numpy()
#img_out = np.moveaxis(img_out, 0, -1)
# print images
ax[0].imshow(img_noisy, cmap='gray')
ax[1].imshow(img_noisy - img_out, cmap='gray')
ax[2].imshow(datasetMRI(PATH_IMG)[0], cmap='gray')
plt.show()
plt.close('all')


### Load CNN - MAYBE CHANGE PATH
""""""