import torch 
import numpy as np 
import parameters
import DNN

def loadModel(model_path):

    model = DNN.Net()
    # model = torch.nn.DataParallel(model) # don't know but has to do with the fact that DataParallel is used during training
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model

def getLipschitzConstants(array, Dj):

    tr = [np.reshape(array[:,i,...], (int(np.sqrt(Dj)) + 1,int(np.sqrt(Dj)) + 1)) for i in range(Dj)]
    
    out = np.zeros(len(tr))
    for i in range(len(tr)):
    
        out[i] = np.max(np.linalg.eigvals(np.matmul(tr[i].T, tr[i])))

    print (np.mean(out), np.max(out), '\n')


if __name__ == "__main__":
    
    #net = loadModel(parameters.Models.DCNN_256_01)
    net = DNN.Net()
    from DNN import init_weights
    net.apply(init_weights)
    D = [64,64,16,16,4,4]

    for j in range(6):

        KERNELS_SIMO = parameters.getNetParameters(net)[1][j].clone().detach().cpu().numpy()

    #print(KERNELS_SIMO.shape)

        getLipschitzConstants(KERNELS_SIMO, D[5-j])
