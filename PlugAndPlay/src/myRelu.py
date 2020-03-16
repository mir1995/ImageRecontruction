import torch

class relu(torch.nn.Module):
    '''
    Implementation of Relu(x - a)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    '''
    
    def __init__(self, alpha):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(relu,self).__init__()

        # initialize alpha
        #if alpha == None:
        #    self.alpha = torch.nn.Parameter(torch.rand(1)) # create a tensor out of alpha
        #else:
        self.alpha = torch.nn.Parameter(alpha) # create a tensor out of alpha
            
        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        
        return torch.nn.functional.relu(x - self.alpha)
