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

BatchSize = 5


seismic_data, data_dim, model_dim, source_amplitudes, x_s, x_r, v_true,vmin,vmax,dx,dt,freq,number_of_shots,omax, omin = DataLoad_Train(device)
# print('source_amplitudes', source_amplitudes.shape, x_s.shape, x_r.shape)
observed_data = seismic_data

# Add SNR noise
#SNR = 10  
#L = 3
#seismic_data = add_noise(seismic_data.detach().cpu(),SNR,L).to(device)

# Add gaussian noise to observed data
#var = torch.tensor(2.3)
#noise = torch.randn(number_of_shots,data_dim[0],data_dim[1])*torch.sqrt(var)
## var = torch.tensor(0.431)
## noise = torch.randn(number_of_shots,data_dim[0],data_dim[1]).mul_(var)
#seismic_data = seismic_data + noise.to(device)
#SNR_signal = torch.sum(observed_data**2)
#SNR_noise = torch.sum(noise**2)
#SNR = 10*torch.log10(SNR_signal/SNR_noise)
#print('SNR is {}'.format(SNR))


plt.figure(figsize=(10.5, 3.5))
plt.imshow(observed_data[5].detach().cpu().T, aspect='auto', cmap='Greys_r',vmin=-0.5, vmax=0.5)
                    # vmin=omin/50, vmax=omax/50)
plt.colorbar()
plt.savefig('observed_data.png')


plt.figure(figsize=(10.5, 3.5))
plt.imshow(seismic_data[5].detach().cpu().T, aspect='auto', cmap='Greys_r',vmin=-0.5, vmax=0.5)
                    # vmin=omin/50, vmax=omax/50)
plt.colorbar()
plt.savefig('observed_data_noise.png')

train_loader = data_utils.DataLoader(MyDataset(seismic_data, source_amplitudes, x_s, x_r, observed_data),
                                     batch_size=BatchSize,
                                     shuffle=False, drop_last=False)

v_init = 1800*torch.ones(100,100).to(device)
# for i_x in range(model_dim[0]):
#     v_init[i_x,:] = v_init[165,:]
v = v_init.clone()
v.requires_grad_()
plt.figure(figsize=(10.5, 3.5))
plt.imshow(v_init.detach().cpu().T, aspect='auto', cmap='RdBu_r',vmin=vmin, vmax=vmax)
plt.savefig('v_init.png')

# Setup optimiser to perform inversion
optimizer = torch.optim.Adam([v], lr=30)
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
mu_ele1 = torch.linspace(1,1000,50)
mu_ele2 = torch.linspace(1000,2000,50)
mu_ele= torch.cat([mu_ele1, mu_ele2])
for col in range(model_dim[0]):
    mu[col,:] = mu_ele


lamda = torch.ones(model_dim[0],model_dim[1])
lamda_ele1 = torch.linspace(2,1,100)
lamda_ele2 = torch.linspace(1,1,40)
lamda_ele3 = torch.linspace(1,2,100)
lamda_ele= torch.cat([lamda_ele1, lamda_ele2,lamda_ele3])
for row in range(model_dim[1]):
   #lamda_ele= torch.cat([torch.linspace(lamda_ele1[row],1,125), torch.linspace(1,lamda_ele1[row],125)])
   lamda_ele= torch.cat([torch.linspace(lamda_ele1[row],1,30), lamda_ele2,torch.linspace(1,lamda_ele3[row],30)])
   lamda[:,row] = lamda_ele

mu = mu*lamda

# lamda = torch.ones(model_dim[0],model_dim[1])
# lamda_ele = torch.linspace(1,1,120)
# # mu_ele2 = torch.linspace(50,100,60)
# # mu_ele= torch.cat([mu_ele1, mu_ele2])
# for col in range(model_dim[0]):
#     lamda[col,:] = lamda_ele


# lamda = torch.ones(model_dim[0],model_dim[1])
# lamda_ele1 = torch.linspace(1,1,110)
# lamda_ele2 = torch.linspace(1,100,30)
# #lamda_ele3 = torch.linspace(1,5,50)

# lamda_ele= torch.cat([lamda_ele1, lamda_ele2])
# for col in range(model_dim[0]):
#     lamda[col,:] = lamda_ele


# for col in range(model_dim[0]):
#     if col > 400:
#         lamda_ele4 = torch.linspace(1,lamda_ele3[col-280],30)
#         lamda_ele_o = torch.cat([lamda_ele1, lamda_ele2*lamda_ele4])
#         lamda[col,:] = lamda_ele_o
    
#     else:
#         lamda[col,:] = lamda_ele
    



DisplayStep = 5
loss_data = []
loss_model_L1 = []
loss_model_L1_portion = []
loss_model_L2 = []
loss_model_RE = []
loss_clean = []
step   = int(number_of_shots/BatchSize)
num_epochs = 400
start  = time.time()

for epoch in range(num_epochs):

    since = time.time()
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

  
        # 那么按照这样计算，当超出界限的速度处会有正的损失并且被加权
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
    #loss1 = 1*TGV(v,lamda.to(device))+0.8*shearlet(v,device)
    loss1 = 0.001*shearlet(v,device)+TGV(v,0.05)
    loss1.backward()
    #grad_tgv = (v.grad).detach().cpu().T - grad_l
    optimizer.step()
    v.data = torch.clamp(v.data, min=1780, max=2100)
    
    
    if epoch == 0:
        loss_dnorm = epoch_loss
    loss_data = np.append(loss_data, loss_loss)
    loss_model_L1 = np.append(loss_model_L1, loss_fm(v, v_true).detach().cpu())
    loss_model_L1_portion = np.append(loss_model_L1_portion, loss_fm(v[20:100,20:230], v_true[20:100,20:230]).detach().cpu())
    loss_model_L2 = np.append(loss_model_L2, loss_fn(v, v_true).detach().cpu())
    loss_model_RE = np.append(loss_model_RE, (loss_fn(v, v_init).detach().cpu())**0.5/(loss_fn(v_true,v_init).detach().cpu())**0.5)
    loss_clean = np.append(loss_clean, loss_cle)
    

    # Print loss and consuming time every epoch
    if (epoch + 1) % 1 == 0:
        # print ('Epoch [%d/%d], Loss: %.10f' % (epoch+1,Epochs,loss.item()))
        # loss1 = np.append(loss1,loss.item())
        print('Epoch: {:d} finished ! Loss: {:.5f}'.format(epoch + 1, epoch_loss / (i+1)))
        time_elapsed = time.time() - since
        print('Epoch: {:d} finished ! Model Loss: {:.5f}'.format(epoch + 1,
                                        loss_fn(v, v_true).detach().cpu()))
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # val
    if (epoch+1) % 20 == 0:
        with torch.no_grad():
           
            plt.figure(figsize=(10.5, 3.5))
            plt.imshow(v.detach().cpu().T, aspect='auto', cmap='RdBu_r',
                     vmin=1700, vmax=2100)
            plt.savefig(str(epoch) + '.png')
            np.savetxt(str(epoch) + '.txt',v.detach().cpu().T)
            
            plt.figure(figsize=(10.5, 3.5))
            plt.imshow(grad_l, aspect='auto', cmap='RdBu_r',
                     vmin=-4, vmax=4)
            plt.savefig(str(epoch) + '_grad_l.png')
            np.savetxt(str(epoch) + '_grad_l.txt',grad_l)
           
           
            # plt.figure(figsize=(10.5, 3.5))
            # plt.imshow(grad_tgv, aspect='auto', cmap='RdBu_r',
            #          vmin=-4, vmax=4)
            # plt.savefig(str(epoch) + '_grad_tgv.png')
            # np.savetxt(str(epoch) + 'grad_tgv.txt',grad_tgv)
           
           


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
np.savetxt('loss_model_L1_portion.txt',loss_model_L1_portion)
np.savetxt('loss_model_L2.txt',loss_model_L2)
np.savetxt('loss_model_RE.txt',loss_model_RE)
np.savetxt('loss_clean.txt',loss_clean)

