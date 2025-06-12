
import torch
from scipy.io import loadmat


        
def shearlet(X,device):   
    filename = '/home/wanghan/test_BP/STFWI/shearlet_n3.mat'
    shearlet = loadmat(filename)
    y = 0
    m,n=X.shape
    for i in range(shearlet['H'].shape[2]):
        shear = torch.tensor(shearlet['H'][:,:,i],dtype=torch.float32).to(device)
        s_X=shear[:m,:n]*torch.fft.fftshift(torch.fft.fft2(X))
        y =y+torch.sum(abs(torch.fft.ifft2(torch.fft.fftshift(s_X))))
    return y
