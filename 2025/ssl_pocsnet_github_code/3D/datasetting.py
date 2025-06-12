import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import scipy.io as sio
class RandomDataset(Dataset):
    def __init__(self, features, masks, mask_uss, length):
        self.features = features
        self.masks = masks
        self.mask_uss = mask_uss
        self.length = length

    def __getitem__(self, index):
        return (torch.Tensor(self.features[index, :]).float(), torch.Tensor(self.masks[index, :]).float(), torch.Tensor(self.mask_uss[index, :]).float())

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