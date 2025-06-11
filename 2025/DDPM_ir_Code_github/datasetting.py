import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import scipy.io as sio

class RandomDataset(Dataset):
    def __init__(self, X_ir, Y, M, MT, length):
        self.X_ir = X_ir
        self.Y = Y
        self.M = M
        self.MT = MT
        self.length = length

    def __getitem__(self, index):
        return (torch.Tensor(self.X_ir[index, :, :]).float(),torch.Tensor(self.Y[index, :, :]).float(),torch.Tensor(self.M[index, :, :]).float(), torch.Tensor(self.MT[index, :, :]))

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