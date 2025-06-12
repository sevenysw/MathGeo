import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import scipy.io as sio

class RandomDataset(Dataset):
    def __init__(self, features, masks, labels, length):
        self.features = features
        self.masks = masks
        self.labels = labels
        self.length = length

    def __getitem__(self, index):
        return (torch.Tensor(self.features[:, :, index]).float(),torch.Tensor(self.masks[:,:, index]).float(),torch.Tensor(self.labels[:, :, index]).float())

    def __len__(self):
        return self.length

def dataload(Filename,Featurename,Labelname):
    f = h5py.File(Filename,'r')
    Features = np.array(f[Featurename])
    Labels = np.array(f[Labelname])
    f.close()
    return Features,Labels,Features.shape[0]

def matload(Filename,dataname):
    data = sio.loadmat(Filename)
    data = data[dataname]
    return data