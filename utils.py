#HELPER FUNCTIONS
import numpy as np
import torch


#Convert an integer label in a one-hot vector
class OneHotEncoder():
    
    def __init__(self, corridor_len):
        self.corridor_len = corridor_len
        
    def __call__(self, sample):
        onehot = np.zeros([len(sample), self.corridor_len])
        tot_places = len(sample)
        onehot[np.arange(tot_places), sample] = 1

        return onehot


#Convert array to tensor
class ToTensor():

  def __call__(self, sample):
    return torch.tensor(sample).float()