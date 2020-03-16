import torch

class groupSort(torch.nn.Module):
    '''
    Implementation of Relu(x - a)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    '''
    
    def __init__(self, _max_ = True, alpha = None, beta = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(groupSort,self).__init__()

        # initialize alpha
        #if alpha == None:
        #    self.alpha = torch.nn.Parameter(torch.rand(1)) # create a tensor out of alpha
        #    self.beta = torch.nn.Parameter(torch.rand(1))
        #else:
        self.alpha = torch.nn.Parameter(alpha) # create a tensor out of alpha
        self.beta = torch.nn.Parameter(beta) 
            
        self.alpha.requiresGrad = True # set requiresGrad to true!
        self.beta.requiresGrad = True # set requiresGrad to true!

        self._max_ = _max_

    def forward(self, x, y):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        if self._max_:
            
            return torch.max(x + self.alpha, y + self.beta)

        else:

            return - torch.max(- (x + self.alpha), -  (y + self.beta))
