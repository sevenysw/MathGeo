import time
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
from data_loader import DataLoad_Train
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from deepwave import scalar
import scipy.io
from scipy.ndimage import gaussian_filter
from shearlet import shearlet
from TGV_diff import TGV
from add_noise import add_noise
from scipy.io import loadmat
import random


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MyDataset(data_utils.Dataset):
    def __init__(self, seismic_data, source_amplitudes, source_locations, receiver_locations, observed_data):

        self.seismic_data = seismic_data
        self.source_amplitudes = source_amplitudes
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations
        self.observed_data = observed_data

    def __getitem__(self, item):

        seismic_data = self.seismic_data[item]
        source_amplitudes = self.source_amplitudes[item]
        source_locations = self.source_locations[item]
        receiver_locations = self.receiver_locations[item]
        observed_data = self.observed_data[item]

        return seismic_data, source_amplitudes, source_locations, receiver_locations, observed_data

    def __len__(self):
        return len(self.seismic_data)

random_seed(13)

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device('cuda:1')
torch.set_default_dtype(torch.float)

BatchSize = 3

filename = "/home/wanghan/test_overthrust/STFWI/shearlet_n3.mat"
shearlet_matrix = loadmat(filename)

seismic_data, data_dim, model_dim, source_amplitudes, x_s, x_r, v_true,vmin,vmax,dx,dt,freq,number_of_shots = DataLoad_Train(device)
# print('source_amplitudes', source_amplitudes.shape, x_s.shape, x_r.shape)
observed_data = seismic_data





train_loader = data_utils.DataLoader(MyDataset(seismic_data, source_amplitudes, x_s, x_r, observed_data),
                                     batch_size=BatchSize,
                                     shuffle=False, drop_last=False)

v_init = (torch.tensor(1/gaussian_filter(1/v_true.cpu().numpy(), 80)).to(device))
# for i_x in range(model_dim[0]):
#     v_init[i_x,:] = v_init[165,:]
v = v_init.clone()
v.requires_grad_()
plt.figure(figsize=(10.5, 3.5))
plt.imshow(v_init.detach().cpu().T, aspect='auto', cmap='RdBu_r',vmin=vmin, vmax=vmax)
plt.savefig('v_init.png')

# Setup optimiser to perform inversion
optimizer = torch.optim.Adam([v], lr=50)
loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fm = torch.nn.L1Loss(reduction='sum')

print()
print('*******************************************')
print('*******************************************')
print('           START TRAINING                  ')
print('*******************************************')
print('*******************************************')
print()

#mu matrix ——这个配置对提升收敛有效果,不加gamma衰减
# mu = torch.ones(model_dim[0],model_dim[1])
# mu_ele = torch.linspace(0.1,0.0005,140)
# for col in range(model_dim[0]):
#     mu[col,:] = mu_ele

#mu matrix
mu = torch.ones(model_dim[0],model_dim[1])
mu_ele1 = torch.linspace(1,1000,90)
mu_ele2 = torch.linspace(1000,30000,40)
mu_ele= torch.cat([mu_ele1, mu_ele2])
for col in range(model_dim[0]):
    mu[col,:] = mu_ele


lamda = torch.ones(model_dim[0],model_dim[1])
lamda_ele1 = torch.linspace(5,1,130)
lamda_ele2 = torch.linspace(1,1,200)
lamda_ele3 = torch.linspace(1,5,130)
lamda_ele= torch.cat([lamda_ele1, lamda_ele2,lamda_ele3])
for row in range(model_dim[1]):
    lamda_ele= torch.cat([torch.linspace(lamda_ele1[row],1,50), lamda_ele2,torch.linspace(1,lamda_ele3[row],50)])
    lamda[:,row] = lamda_ele

mu = mu*lamda


gamma = torch.ones(model_dim[0],model_dim[1])
gamma_ele1 = torch.linspace(1,30,150)
gamma_ele2 = torch.linspace(30,1,150)
gamma_ele= torch.cat([gamma_ele1, gamma_ele2])
for col in range(model_dim[1]):
    gamma[:,col] = gamma_ele


# lamda = torch.ones(model_dim[0],model_dim[1])
# ele1 = torch.linspace(1,5,140)
# ele2 = torch.linspace(5,50,140)
# for i in range(model_dim[1]):
#     lamda_ele1 = torch.linspace(ele1[i],ele2[i],165)
#     lamda_ele2 = torch.linspace(ele2[i],ele1[i],165)
#     lamda_ele= torch.cat([lamda_ele1, lamda_ele2])
#     lamda[:,i] = lamda_ele

# lambbda = torch.ones(model_dim[0],model_dim[1])
# lambbda_ele1 = torch.linspace(1,2,70)
# lambbda_ele2 = torch.linspace(2,5,70)
# #lambbda_ele3 = torch.linspace(1,1,70)

# lambbda_ele= torch.cat([lambbda_ele1, lambbda_ele2])
# for col in range(model_dim[0]):
#     lambbda[col,:] = lambbda_ele

# lamda = torch.mul(lamda,lambbda)


    



DisplayStep = 8
loss_data = []
loss_model_L1 = []
loss_model_L2 = []
loss_model_RE = []
loss_clean = []
step   = int(number_of_shots/BatchSize)
num_epochs = 10
start  = time.time()

for epoch in range(num_epochs):


    epoch_loss = 0.0
    loss_loss = 0.0
    loss_cle = 0.0
    optimizer.zero_grad()
    #gamma = (epoch+500)/500*mu
    #sigma = (epoch+1000)/1000*lamda

    for i, (s_d_i, s_a_i, x_s_i, x_r_i, o_d_i) in enumerate(train_loader):

        iteration = epoch * step + i + 1

        # (num_shots, 1, data_dim[0] * data_dim[1]) ny = 2301 nx = 751

        # Forward prediction
        out = scalar(
            v, dx, dt,
            source_amplitudes=s_a_i.to(device),
            source_locations=x_s_i.to(device),
            receiver_locations=x_r_i.to(device),
            pml_freq=freq,
            accuracy=8,
        )[-1]

        # predicted = out
        # predicted_norm = (predicted.detach() ** 2).mean().sqrt().item()
        # if predicted_norm > 0:
        #     normed_predicted = predicted / predicted_norm
        # else:
        #     normed_predicted = predicted

        # loss = (
        #     loss_fn(normed_predicted.to(device),
        #             s_d_i.to(device))
        # )

       
        loss = loss_fn(out.to(device), s_d_i.to(device)) 
        
        #loss = TV(v,1)#+TGV(v,1e-1,1e-1)# +gamma*shearlet(v,device) + (((v - 1200) ** 2) * lower_mask).sum() + (((v - 6000) ** 2) * upper_mask).sum()
       
#        if np.isnan(float(loss.item())):
#            raise ValueError('loss is nan while training')
       

        epoch_loss += loss.item()
        # Loss backward propagation
       # loss1.backward()
        loss.backward()
        
        loss_loss = loss_loss+loss_fn(out.to(device), s_d_i.to(device)).item()
        loss_cle = loss_cle+loss_fn(out.to(device), o_d_i.to(device)).item()


        # Print loss
#        if iteration % DisplayStep == 0:
#            print('Epoch: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'.format(epoch + 1,num_epochs,
#                                                                                   iteration,step * num_epochs,
#                                                                                   loss.item()))

    grad_l = (mu.to(device)*v.grad).detach().cpu().T
    v.grad = v.grad * mu.to(device)
    loss1 = 8*TGV(v,1)+1*shearlet(v,shearlet_matrix,device)
    #loss1 = 2*shearlet(v,device)
    loss1.backward()
    #grad_tgv = (v.grad).detach().cpu().T - grad_l
    optimizer.step()
    v.data = torch.clamp(v.data, min=2400, max=5700)
    
    
    if epoch == 0:
        loss_dnorm = epoch_loss
    loss_data = np.append(loss_data, loss_loss)
    loss_model_L1 = np.append(loss_model_L1, loss_fm(v, v_true).detach().cpu())
    loss_model_L2 = np.append(loss_model_L2, loss_fn(v, v_true).detach().cpu())
    loss_model_RE = np.append(loss_model_RE, (loss_fn(v, v_init).detach().cpu())**0.5/(loss_fn(v_true,v_init).detach().cpu())**0.5)
    loss_clean = np.append(loss_clean, loss_cle)
    

    # Print loss and consuming time every epoch
#    if (epoch + 1) % 1 == 0:
#        # print ('Epoch [%d/%d], Loss: %.10f' % (epoch+1,Epochs,loss.item()))
#        # loss1 = np.append(loss1,loss.item())
#        print('Epoch: {:d} finished ! Loss: {:.5f}'.format(epoch + 1, epoch_loss / (i+1)))
#
#        print('Epoch: {:d} finished ! Model Loss: {:.5f}'.format(epoch + 1,
#                                        loss_fn(v, v_true).detach().cpu()))


    # val
#    if (epoch+1) % 20 == 0:
#        with torch.no_grad():
#           
#            plt.figure(figsize=(10.5, 3.5))
#            plt.imshow(v.detach().cpu().T, aspect='auto', cmap='RdBu_r',
#                     vmin=vmin, vmax=vmax)
#            plt.savefig(str(epoch) + '.png')
#            np.savetxt(str(epoch) + '.txt',v.detach().cpu().T)
#            
#            plt.figure(figsize=(10.5, 3.5))
#            plt.imshow(grad_l, aspect='auto', cmap='RdBu_r',
#                     vmin=-4, vmax=4)
#            plt.savefig(str(epoch) + '_grad_l.png')
#            np.savetxt(str(epoch) + '_grad_l.txt',grad_l)
#           
#           
#            plt.figure(figsize=(10.5, 3.5))
#            plt.imshow(grad_tgv, aspect='auto', cmap='RdBu_r',
#                     vmin=-4, vmax=4)
#            plt.savefig(str(epoch) + '_grad_tgv.png')
#            np.savetxt(str(epoch) + 'grad_tgv.txt',grad_tgv)
           
           


# Record the consuming time
time_elapsed = time.time() - start
np.savetxt('time_cost.txt',[time_elapsed])
print('Training complete in {:.0f}m  {:.0f}s' .format(time_elapsed //60 , time_elapsed % 60))

#data = {}
#data['loss_model'] = loss_model
#scipy.io.savemat('12_ModelLoss.mat', data)
#
#data = {}
#data['loss_data'] = loss_data
#scipy.io.savemat('12_DataLoss.mat', data)

np.savetxt('loss_data.txt',loss_data)
np.savetxt('loss_model_L1.txt',loss_model_L1)
np.savetxt('loss_model_L2.txt',loss_model_L2)
np.savetxt('loss_model_RE.txt',loss_model_RE)
np.savetxt('loss_clean.txt',loss_clean)

