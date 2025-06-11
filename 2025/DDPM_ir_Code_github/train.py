import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from models import *
from utils import *

from argparse import ArgumentParser
import datasetting

import random
import logging
import os
import scipy.io as sio
import math

random.seed(5267)


parser = ArgumentParser(description='DPM Nonuniform Interpolation')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=500, help='epoch number of end training')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
parser.add_argument('--timesteps', type=int, default=1000)
parser.add_argument('--losstype', type=str, default="l2")
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
batch_size = args.batch_size
learning_rate = args.learning_rate
timesteps = args.timesteps
losstype = args.losstype
gpu_list = args.gpu_list
model_dir = "./%s/Model_DPM_learning_lr_%.4f" % (args.model_dir, learning_rate)

log_file_name = "./%s/Log_DPM_learning_lr_%.4f_.txt" % (args.log_dir, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

print("Conditional diffusion model based seismic data interpolation")
print("nonuniform irregular interpolation")
print("The parameters are:"+str(args))
# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

print('------------------------------------Data Set Loading...-------------------------------------')
## load X
data_filename = 'X_ir_train.mat'
filepath = '%s/%s'%(args.data_dir,data_filename)
dataname = 'X_ir_train'
X = datasetting.matload(filepath,dataname)
ntrain = X.shape[0]

## load Y
data_filename = 'Y_train.mat'
filepath = '%s/%s'%(args.data_dir,data_filename)
dataname = 'Y_train'
Y = datasetting.matload(filepath,dataname)

# load Pos
data_filename = 'M_train.mat'
filepath = '%s/%s'%(args.data_dir,data_filename)
dataname = 'M_train'
M = datasetting.matload(filepath,dataname)

# load Pos_T
data_filename = 'MT_train.mat'
filepath = '%s/%s'%(args.data_dir,data_filename)
dataname = 'MT_train'
MT = datasetting.matload(filepath,dataname)

print('The number of training pairs:%d'%ntrain)

print("  =========== Training Begin ===========   ")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('The modle is training based on GPU:%s'%gpu_list)
else:
    print('The modle is training based on CPU')


rand_loader = DataLoader(dataset=datasetting.RandomDataset(X, Y, M, MT, ntrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

model = Unet(
    dim=64,
    channels=2,
    dim_mults=(1, 2, 4,8),
    out_dim=1
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

Loss_epoch = []
if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))
    filepath = "./%s/Loss_training_epoch_%d_lr_%.4f.mat"%(args.log_dir,start_epoch,learning_rate)
    Loss = sio.loadmat(filepath)
    Loss_epoch = Loss["Loss_epoch_training"]
    Loss_epoch = Loss_epoch.tolist()[0]

print('--------------------------------------Model Training...--------------------------------------')

for epoch_i in range(start_epoch+1, end_epoch+1):
    Loss_all = 0
    for (x, y, m, mt) in rand_loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        mt = mt.to(device)
        batch_size = x.shape[0]
        x = x.view(-1,1,32,64)
        y = y.view(-1,1,64,64)
        m = m.view(-1,1,64,32)
        mt = mt.view(-1,1,32,64)
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # Compute and print loss
        loss_all = p_losses(model, y, x, m, mt, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type=losstype)

        # loss_all = loss_discrepancy
        Loss_all = Loss_all + loss_all.item()
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        output_data = "[%02d/%02d] Total Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item())
        print(output_data)
    Loss_epoch.append(Loss_all / math.ceil(ntrain/batch_size))
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 50 == 0:
        sio.savemat("./%s/Loss_training_epoch_%d_lr_%.4f.mat"%(args.log_dir,epoch_i,learning_rate),{"Loss_epoch_training":Loss_epoch})
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters


