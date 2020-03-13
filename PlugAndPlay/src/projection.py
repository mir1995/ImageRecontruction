import torch 
import numpy as np

def projD(array):

    # (64, 1, 9, 9)
    # would we need to map the numpy back to the CPU??
    # c = output.cpu().data.numpy()

    npA = array
    
    npA = np.fft.fft2(npA)
    

    norms = np.maximum(1, np.sqrt(np.sum(np.absolute(npA)**2, axis = 0))) # i am assumig we are changing only the ones which are not the unit ball
    
    #norms = np.sqrt(np.sum(np.absolute(npA)**2, axis = 0))
    #print(norms.shape)
    norms = np.expand_dims(norms,1)
    #print(norms.shape)
    normsM = np.repeat(norms, npA.shape[0], axis = 0)  
   

    npA = npA/normsM  # normsM  ?  should check this?

    # NOT GREAT AT ALL THIS CODE
    return np.real(np.fft.ifft2(npA))

def algorithm1(hbar_tensor, N, Tensor):
    """
        WARNING: When converting from a array
        on a GPU to a numpy array on a CPU 
        we are loosing the tracking on the computational
        graph. Make sure it is enough to bring everything 
        back to GPU.
    """
    hbar = hbar_tensor.clone().detach().cpu().numpy()

    epsilon = np.random.uniform(low=0.001, high=1)
    k_t = np.random.randn(*hbar.shape)
    for _ in range(N):
        h_t = k_t # projC(k_t) I don't know how is this needed
        lambdat = np.random.uniform(low=epsilon, high= 2 - epsilon) # it appers in the paper that they vary epsilon at each iteration
        k_t = k_t + lambdat*(projD((2*h_t - k_t + hbar)/2) - h_t) # this won't be affected if i do not do the projection onto C although change code eventually
    
    #print(k_t.size())
    
    return torch.as_tensor(torch.tensor(h_t, requires_grad=True), device=torch.device('cpu')).type(Tensor) # should it not be k_t




if __name__ == "__main__":
    
    a = torch.rand(2,1,3,3)
    #print(algorithm1(a,10).size())