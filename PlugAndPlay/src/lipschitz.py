import torch 
import numpy as np 
import parameters
import DNN

def loadModel(model_path):

    model = DNN.Net()
    # model = torch.nn.DataParallel(model) # don't know but has to do with the fact that DataParallel is used during training
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model

def getLipschitzConstants(array):

    tr = [np.reshape(array[i,...], (9,9)) for i in range(64)]

    tr = [np.max(np.linalg.eigvals(np.matmul(m.T, m))) for m in tr]

    print (*tr)


if __name__ == "__main__":
    
    net = loadModel(parameters.Models.DCNN_256_001)

    KERNELS_SIMO = parameters.getNetParameters(net)[0][0].clone().detach().cpu().numpy()

    print(KERNELS_SIMO.shape)

    getLipschitzConstants(KERNELS_SIMO)