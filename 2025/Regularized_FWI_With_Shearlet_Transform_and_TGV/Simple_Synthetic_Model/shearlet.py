
import torch
from scipy.io import loadmat


        
def shearlet(X,device):   
    filename = "/home/wanghan/SA_model/STFWI/matlab_SAmodel.mat"
    shearlet = loadmat(filename)
    y = 0
    m,n=X.shape
    for i in range(shearlet['H'].shape[2]):
        shear = torch.tensor(shearlet['H'][:,:,i],dtype=torch.float32).to(device)
        s_X=shear[:m,:n]*torch.fft.fftshift(torch.fft.fft2(X))
        y =y+torch.sum(abs(torch.fft.ifft2(torch.fft.fftshift(s_X))))
    return y
    # Z = torch.zeros(m,m-n).to(device)
    # X = torch.cat([D,Z],1)
    # for i in range(shearlet['H'].shape[2]):
    #     shear = torch.tensor(shearlet['H'][:,:,i],dtype=torch.float32).to(device)
    #     s_X=shear*torch.fft.fftshift(torch.fft.fft2(X))
    #     y =y+torch.sum(abs(torch.fft.ifft2(torch.fft.fftshift(s_X))))
    # return y
