# %% [markdown]
# # Import libs

# %% [markdown]
# Code for seismic registration
# $A$ and $B$ are PP and PS seismic data
# $F(.,u)$ is a forward operator for seismic phase change $u$
# $$F(A,u(t,x)) =A(t-u(t,x),x) $$
# $$\hat z = \argmin_z \|F(A,u(z;\theta)) - B\|_2^2$$
# Then $u = u(\hat z)$
# The most key work is programming F. Others exist.

# %% [markdown]
# Test with different data, shift, noise level, iteration.

# %%
from __future__ import print_function
import matplotlib.pyplot as plt


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *
import time
import torch
import torch.optim
import argparse
import platform
import random
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import structural_similarity
from utils.denoising_utils import *
from utils.measure import compute_SNR
import scipy.io as io
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

is_cuda = True
if is_cuda:
    dtype = torch.cuda.FloatTensor
    itype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    itype = torch.LongTensor
imsize =-1
figsize = 5 
PLOT = True


parser = argparse.ArgumentParser()

parser.add_argument('--idata', type=int, default=3)
parser.add_argument('--sigma', type=float, default=0.2)
parser.add_argument('--ishs', type=int, default=2)
parser.add_argument('--num_iter', type=int, default=5000)
parser.add_argument('--amp', type=int, default=140)
args = parser.parse_args()

sigma_ = args.sigma
amp = args.amp

if platform.system().lower() == 'windows':
    fn_root = '/home/huyue/Torch/DLR/results/pc_exp_idata_'+str(args.idata)+'_ishs_'+str(args.ishs)+'_sigma_'+str(args.sigma)+'_amp_'+str(args.amp)+'/'
else:
    fn_root = '/home/huyue/Torch/DLR/results/ex_idata_'+str(args.idata)+'_ishs_'+str(args.ishs)+'_sigma_'+str(args.sigma)+'_amp_'+str(args.amp)+'/'
if not os.path.exists(fn_root):
    os.mkdir(fn_root)

# %% [markdown]
# Load data

# %%
vmax_d = 1.0
vmin_d = -vmax_d 
ylabel='Depth (m)'
# data 1
if args.idata == 0:
    fname = '/home/huyue/Torch/DLR/data/registration/sigmoid.mat'
    d = io.loadmat(fname)['D']
    A0,A1,A2 = amp, amp, amp
    freq = 2 # angle freqency
elif args.idata == 1:
    # data 2
    d = io.loadmat('/home/huyue/torch/DLR/data/registration/seismic.mat')['data'][0][1][:200,:256]/255-0.5
    d = d * 2
    A0,A1,A2 = amp, amp, amp
    freq = 3
if args.idata == 2:
    fname = '/home/huyue/torch/DLR/data/registration/F3.mat'
    d = io.loadmat(fname)['D'][0:400,0:800]         #[200:400,400:656]
    A0,A1,A2 = amp, amp, amp
    freq = 6
if args.idata == 3:
    fname = '/home/huyue/Torch/DLR/data/registration/Kerry.mat'
    d = io.loadmat(fname)['D'][0:1200,0:680]
    A0,A1,A2 = amp, amp, amp
    freq = 10
    ylabel='Time (ms)'
# plt.imshow(d,vmin=-0.5,vmax=0.5,cmap=plt.cm.gray)
# plt.colorbar()
label_d = 'Amplitude'
plot_image_grid([d[None,:]], factor=figsize, nrow=1,fn = fn_root+'original_data',vmin=vmin_d,vmax=vmax_d,clabel = label_d,ylabel=ylabel)

# %%
# generator a phase

dx, dz = 4,4
(nz, nx) = d.shape
sh = np.zeros((nz,nx))
cz = [int(nz/2 - nz/4), int(nz/2 + nz/4)]
cx = [int(nx/2), int(nx/2)]
label_s = 'Shift (m)'
(xv,zv) = np.meshgrid(np.linspace(1,nx,nx),np.linspace(1,nz,nz))

betaz = 33*nz/200
betax = 33*2.5*nx/256
shs = 0
if args.ishs == 0:
# case 1: two circles
    sgn = np.array([1,-1])
    for i in range(2):
        shs += np.exp(-(((zv - cz[i])/betaz )**2+((xv-cx[i])/betax)**2))*A0*sgn[i]/dz
elif args.ishs == 1:
# case 2:line
    shs = (zv / nz * 2*A1-A1) / dz
elif args.ishs == 2:
# case 3: sin
    shs = A2 * np.sin(zv / nz * freq * np.pi)/dz
    label_s = 'Shift (ms)'

vmin_s = np.min(shs)*dx
vmax_s = np.max(shs)*dx

# plt.figure()
# plt.imshow(shs)
# plt.colorbar()
plot_image_grid([dx*shs[None,:]], factor=figsize, nrow=1,fn = fn_root+'original_shift',vmin=vmin_s,vmax=vmax_s,clabel = label_s, cmap='jet',ylabel=ylabel)

# %% [markdown]
# $I_2 = I_1\left[ z+u(z,x),x\right]$

# %%
def apply_shift_torch_mat2(d,shs):
    (nc,nz,nx) = d.shape
    dshift = d * 0
    ind0 = torch.linspace(0,nz-1,nz).type(itype)
    t1 = time.time()
    for j in range(nx):
        ap = shs[0,:,j] - torch.floor(shs[0,:,j])
        ind1 = torch.floor(ind0 + shs[0,:,j]).type(itype)
        ind2 = ind1 + 1
        ind2[ind1<0] = 0
        ind1[ind1<0] = 0
        ind2[ind1>nz-2] = nz - 1
        ind1[ind1>nz-2] = nz - 1
        # ind2 = ind1 + 1
        # ap = ind0 + shs[0,:,j] - ind1
        dshift[0,ind0,j] = (1-ap)*d[0,ind1,j] + ap*d[0,ind2,j]
    print(time.time()-t1)
    return dshift


def apply_shift_torch_mat(d,shs):
    (nc,nz,nx) = d.shape
    dshift = d * 0
    ind0 , c_ = torch.meshgrid(torch.linspace(0,nz-1,nz),torch.linspace(0,nx-1,nx))
    ind0 = ind0.type(itype)
    c_ =   c_.type(itype)
    # t1 = time.time()
    # for j in range(nx):
    ap = shs - torch.floor(shs)
    ind1 = torch.floor(ind0 + shs).type(itype)
    ind2 = ind1 + 1
    ind2[ind1<0] = 0
    ind1[ind1<0] = 0
    ind2[ind1>nz-2] = nz - 1
    ind1[ind1>nz-2] = nz - 1
    # ind2 = ind1 + 1
    # ap = ind0 + shs[0,:,j] - ind1
    ind0r = torch.reshape(ind0,(nz*nx,1))
    ind1r = torch.reshape(ind1,(nz*nx,1))
    ind2r = torch.reshape(ind2,(nz*nx,1))
    c_r   = torch.reshape(c_,  (nz*nx,1))
    dshift[0,ind0r,c_r] = (1-ap[0,ind0r,c_r ])*d[0,ind1r,c_r ] + ap[0,ind0r,c_r]*d[0,ind2r,c_r ]
    # for j in range(nx):
    #     dshift[0,ind0[:,j],j] = (1-ap[0,:,j])*d[0,ind1[0,:,j],j] + ap[0,:,j]*d[0,ind2[0,:,j],j]
    # print(time.time()-t1)
    return dshift


d_torch = np_to_torch(d).type(dtype)
shs_torch=np_to_torch(shs).type(dtype)

dshift0 = apply_shift_torch_mat(d_torch,shs_torch)
dshift_noise = dshift0 + torch.randn(d_torch.shape).type(dtype) * sigma_
dshift = torch_to_np(dshift0)
# plt.imshow(dshift)
# plt.colorbar()

plot_image_grid([torch_to_np(dshift_noise)[None,:]], factor=figsize, nrow=1,fn = fn_root+'shift_data',vmin=vmin_d,vmax=vmax_d,clabel = label_d,ylabel=ylabel)

# %% [markdown]
# # Setup

# %%
INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 100
show_every_bt = 100
exp_weight=0.99


num_iter = args.num_iter
input_depth = 1

class EarlyStop():
    def __init__(self, size, patience):
        self.patience = patience
        self.wait_count = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.img_collection = []
        self.stop = False
        self.size = size

    def check_stop(self, current, cur_epoch):
      #stop when variance doesn't decrease for consecutive P(patience) times
        if current < self.best_score:
            self.best_score = current
            self.best_epoch = cur_epoch
            self.wait_count = 0
            should_stop = False
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience
        return should_stop

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.size:
            self.img_collection.pop(0)

    def get_img_collection(self):
        return self.img_collection

def myMetric(x1, x2):
    return ((x1 - x2) ** 2).sum() / x1.size
    # return (np.abs(x1 - x2)).sum() / x1.size


# net = skip(
#             input_depth, input_depth, 
#             num_channels_down = [8, 16, 32], 
#             num_channels_up   = [8, 16, 32],
#             num_channels_skip = [0, 0, 4], 
#             upsample_mode='bilinear',
#             need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU')


net = skip(
            input_depth, input_depth, 
            num_channels_down = [8, 16, 32, 64, 128], 
            num_channels_up   = [8, 16, 32, 64, 128],
            num_channels_skip = [0, 0, 0, 4, 4], 
            upsample_mode='bilinear',
            need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU')
# print(net)
net = net.type(dtype)
    
net_input = get_noise(input_depth, INPUT, (d.shape[0], d.shape[1])).type(dtype).detach()

# vise_graph = torchviz.make_dot(net(net_input).mean(), params=dict(list(net.named_parameters()) + [('x', net_input)]))
# vise_graph.view()

print(net_input.shape)
# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

def tv_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.nn.ConstantPad2d((0,0,1,0),0) (torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2))
     tv_w = torch.nn.ConstantPad2d((1,0,0,0),0) (torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2))
     return weight*torch.sqrt(tv_h+tv_w+0.000001).sum()/(bs_img*c_img*h_img*w_img)

# %%
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
shift_avg = None
last_net = None
srn_noisy_last = 0
s1 = []
s2 = []
s3 = []
s4 = []
loss_ = []
PLOT = True
d_torch = np_to_torch(d).type(dtype)
i = 0
t1 = time.time()

buffer_size = 100
patience = 700
variance_history = []
x_axis = []
earlystop = EarlyStop(size=buffer_size,patience=patience)

def closure():
    
    global i, out_avg, srn_noisy_last, last_net, net_input, s1,s2,s3,s4, loss_, shift_avg,t1
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    out = net(net_input)
    
    zerot = torch.zeros(net_input.shape).cuda()
    # print(out.shape) 
    
    dshift_pred = apply_shift_torch_mat(d_torch,out[0])   

        # Smoothing
    if out_avg is None:
        shift_avg = dshift_pred.detach()
        out_avg   = out[0].detach()
    else:
        shift_avg = shift_avg * exp_weight + dshift_pred.detach() * (1 - exp_weight)
        out_avg   = out_avg * exp_weight + out[0].detach() * (1 - exp_weight)

    # tv_loss_val = tv_loss(out, 0.02) #0.02
    total_loss = mse(dshift_pred, dshift_noise)  # + 0.1 * mse(net_input, zerot )
    total_loss.backward()
        
    srn_noisy = compute_SNR(d, dshift) 
    srn_gt    = compute_SNR(dshift_pred.detach().cpu().numpy()[0], dshift) 
    srn_gt_sm = compute_SNR(shift_avg.detach().cpu().numpy()[0],dshift) 
    srn_shift = compute_SNR(out_avg.cpu().numpy()[0],shs) 
    current_time = time.time() - t1
    s1.append(srn_noisy)
    s2.append(srn_gt)
    s3.append(srn_gt_sm)
    s4.append(srn_shift)
    loss_.append(total_loss.item())
    r_img_np = dshift_pred.detach().cpu().numpy()
    #variance hisotry
    r_img_np = r_img_np.reshape(-1)
    earlystop.update_img_collection(r_img_np)
    img_collection = earlystop.get_img_collection()
    if len(img_collection) == buffer_size:
        ave_img = np.mean(img_collection,axis = 0)
        variance = []
        for tmp in img_collection:
            variance.append(myMetric(ave_img, tmp))
        cur_var = np.mean(variance)
        cur_epoch = i
        variance_history.append(cur_var)
        x_axis.append(cur_epoch)
        if earlystop.stop == False:
            earlystop.stop = earlystop.check_stop(cur_var, cur_epoch)

    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print ('Iteration %05d    SNR_shift: %f   SRN_gt: %f SNR_gt_sm: %f Time: %f Loss %f ' % (i,  srn_shift, srn_gt, srn_gt_sm, current_time,total_loss.item() ), '\r', end='')
    
    if  PLOT and i % show_every == 0:
        plot_image_grid([torch_to_np(dx*out_avg)[None,:]], factor=figsize, nrow=1,fn = fn_root+'_shift_iter_%05d'%i, vmin=vmin_s,vmax=vmax_s,clabel = label_s, title = "Predicted Shift SNR:%.2f dB"%srn_shift, cmap='jet')
        plot_image_grid([torch_to_np(shift_avg)[None,:]], factor=figsize, nrow=1,fn = fn_root+'_shift_data_iter_%05d'%i ,vmin=vmin_d,vmax=vmax_d,clabel = label_d, title = "Predicted Data SNR:%.2f dB"%srn_gt_sm)
        
    # Backtracking
    if i % show_every_bt:
        if srn_noisy - srn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            srn_noisy_last = srn_noisy
            
    if i==args.num_iter-1:
        fn = fn_root+'iter_%05d'%i
        np.savez(fn,shs_pred=torch_to_np(out_avg),dshs_pred=torch_to_np(dshift_pred),time = current_time, snr_s=s4, snr_d = s3, loss=loss_,t=earlystop.best_epoch,var=variance_history,x_axis=x_axis)

    # if i==earlystop.best_epoch:
    #     fn = fn_root+'iter_%05d'%i
    #     np.savez(fn+'0',shs_pred=torch_to_np(out_avg),dshs_pred=torch_to_np(dshift_pred))

    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


# # %%
# # plt.plot(s1,label='srn_data_base')
# # plt.plot(s2,label='srn_gt')
# plt.figure()
# plt.plot(s3,label='Data')
# plt.plot(s4,label='Shift')
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('SNR (dB)')

# plt.savefig(fn_root+'snr.png')
# plt.close()
# # plt.savefig(fn_root+'snr.pdf')
# # %%
# # q = plot_image_grid([np.clip(torch_to_np(shift_avg)[None,:], -1, 1),dshift[None,:]], factor=18);

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('shiftSNR', color=color)
ax1.plot(s4, color=color)
ax1.tick_params(axis='y', labelcolor=color)


ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Variance', color=color)
ax2.plot(x_axis, variance_history, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.title('ES-WMV')
plt.axvline(x=earlystop.best_epoch, label='detection',color='y')
plt.legend()
plt.show()

plt.savefig( '/home/huyue/torch/DLR/exam_04.png') 

