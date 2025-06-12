from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import time
import models
import datasetting
import utils

parser = ArgumentParser(description='POCS-Net')

parser.add_argument('--epoch', type=int, default=50, help='epoch number for prediction')
parser.add_argument('--layer_num', type=int, default=30, help='phase number of POCS-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--model_dir', type=str, default='model_pre&post', help='trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='prediction data directory')
parser.add_argument('--save_dir', type=str, default='prediction_result_30', help='save directory for results')
args = parser.parse_args()

epoch = args.epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
gpu_list = args.gpu_list
model_dir = "./%s/Model_3DPOCSNet_learning_lr_%.4f_layernum_%02d" % (args.model_dir, learning_rate,layer_num)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('------------------------------------3D POCS-Net Setting-------------------------------------')
print('Epoch:%d'%epoch)
print('Learning rate:%.4f'%learning_rate)
print('Data directory:%s'%args.data_dir)
print('Model directory:%s'%args.model_dir)
print('Save directory:%s'%args.save_dir)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('The modle is predicting based on GPU:%s'%gpu_list)
else:
    print('The modle is predicting based on CPU')
print('------------------------------------Data Set Loading...-------------------------------------')
##load original data
data_filename = 'X3Dsyn.mat'
filepath = '%s/%s'%(args.data_dir,data_filename)
dataname = 'D'
True_data = datasetting.matload(filepath,dataname)
print('The size of the data is: ',True_data.shape)
True_data = True_data.transpose(2,1,0)
m = True_data.shape[0]
n = True_data.shape[1]
p = True_data.shape[2]
len_data = m*n*p

##load sampling matrix
phi_filename = 'Phi_X3Drandom0.5.mat'
filepath = '%s/%s'%(args.data_dir,phi_filename)
dataname = 'Phi'
Phi = datasetting.matload(filepath,dataname)
Phi = Phi.transpose(2,1,0)
print('--------------------------------------Model Loading...--------------------------------------')
model = models.POCSNet_prediction(layer_num,m,n,p,len_data)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch)))
print("Total number of param in 3DPOCSNet is ", sum(x.numel() for x in model.parameters()))

print('-------------------------------------Model Prediction...------------------------------------')

with torch.no_grad():
    start_time = time.time()
    x = True_data / np.max(True_data)
    x = torch.from_numpy(x).type(torch.FloatTensor)
    x = x.to(device)
    x= x.view(1,1,m,n,p)
    Phi = torch.from_numpy(Phi).type(torch.FloatTensor)
    Phi = Phi.to(device)
    Phi = Phi.view(1,1,m,n,p)
    b = torch.mul(Phi,x)
    [x_output] = model(Phi,b,b)
    x_output = x_output.view(m,n,p)
    end_time = time.time()
    compute_time = round(end_time - start_time, 2)
    Original_data =  True_data.transpose(2,1,0)

    Missing_data =  True_data * Phi.view(m,n,p).cpu().numpy()
    Missing_data = Missing_data.transpose(2,1,0)

    Prediction_data =  x_output * np.max(True_data)
    Prediction_data = Prediction_data.cpu().data.numpy()
    Prediction_data = Prediction_data.transpose(2,1,0)

snr = utils.SNR(Original_data,Prediction_data)
print("Computation time:%.2f,SNR:%.2fdB"%(compute_time,snr))
save_dir = '%s/%s_%s_Prediction_data.mat'%(args.save_dir,data_filename[0:-4],phi_filename[0:-4])
sio.savemat(save_dir,{'Prediction_data':Prediction_data,'Original_data':Original_data,'Missing_data':Missing_data,'SNR':snr,'compute_time':compute_time})








