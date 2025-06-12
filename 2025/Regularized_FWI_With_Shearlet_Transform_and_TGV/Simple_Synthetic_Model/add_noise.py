import torch
import numpy as np



def add_noise(observed_data,SNR,L):
    [num_shots,nr,nt] = observed_data.shape
    for i in range (num_shots):
        din = observed_data[i,:,:].T
        h = torch.tensor(np.hamming(L))
        noise = torch.randn(nt,nr)
        noise = torch.tensor(np.apply_along_axis(lambda m: np.convolve(m, h, mode='full'), axis=1, arr=noise.T)).T
        noise = noise[int(np.floor((L-1)/2)-1):(int(np.floor((L-1)/2))+nt-1),:]
        alpha = torch.sqrt((din**2).sum()/(SNR*noise**2).sum())
        noise_add = alpha*noise
        dout = din+noise_add
        observed_data[i,:,:] = dout.T
    return observed_data