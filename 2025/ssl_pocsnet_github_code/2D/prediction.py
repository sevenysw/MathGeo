from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
import scipy.io as sio
import time
import models
import datasetting
import utils

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = ArgumentParser(description='POCS-Net')

parser.add_argument('--epoch', type=int, default=500, help='epoch number for prediction')
parser.add_argument('--layer_num', type=int, default=35, help='phase number of POCS-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gpu_list', type=str, default='1', help='gpu index')
parser.add_argument('--model_dir', type=str, default='model', help='trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='prediction data directory')
parser.add_argument('--save_dir', type=str, default='prediction_result_35', help='save directory for results')
args = parser.parse_args()

epoch = args.epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
gpu_list = args.gpu_list
model_dir = "./%s/Model_2DPOCSNet_learning_lr_%.4f_layernum_%02d" % (args.model_dir, learning_rate,layer_num)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('------------------------------------2D POCS-Net Setting-------------------------------------')
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
data_filename = 'slice400.mat'
filepath = '%s/%s'%(args.data_dir,data_filename)
dataname = 'slice'
True_data = datasetting.matload(filepath,dataname)
print('The size of the data is: ',True_data.shape)
True_data= torch.from_numpy(True_data).type(torch.FloatTensor)
True_data = True_data.to(device)
True_data = torch.transpose(True_data,0,1)
m = True_data.shape[0]
n = True_data.shape[1]
len_data = m*n
True_data = True_data.view(1,1,m,n)

##load sampling matrix
phi_filename = 'mask400.mat'
filepath = '%s/%s'%(args.data_dir,phi_filename)
dataname = 'mask'
Phi = datasetting.matload(filepath,dataname)
Phi= torch.from_numpy(Phi).type(torch.FloatTensor)
Phi = Phi.to(device)
Phi = torch.transpose(Phi,0,1)
Phi = Phi.view(1,1,m,n)
print('--------------------------------------Model Loading...--------------------------------------')
model = models.POCSNet_prediction(layer_num,m,n,len_data)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch)))
print("Total number of param in 2DPOCSNet is ", sum(x.numel() for x in model.parameters()))

print('-------------------------------------Model Prediction...------------------------------------')

with torch.no_grad():
    start_time = time.time()
    x = True_data / torch.abs(True_data).max()
    b = torch.mul(Phi,x)
    x0 = b
    [x_output] = model(Phi,b,x0)
    x_output = x_output.view(1,1,m,n)
    end_time = time.time()
    compute_time = round(end_time - start_time, 2)
    Original_data =  True_data.view(m,n)
    Original_data = torch.transpose(Original_data,0,1)
    Original_data = Original_data.cpu().numpy()

    Missing_data =  x0.view(m,n) * torch.abs(True_data).max()
    Missing_data = torch.transpose(Missing_data,0,1)
    Missing_data = Missing_data.cpu().numpy()

    Prediction_data =  x_output.view(m,n) * torch.abs(True_data).max()
    Prediction_data = torch.transpose(Prediction_data,0,1)
    Prediction_data = Prediction_data.cpu().data.numpy()

snr = utils.SNR(Original_data,Prediction_data)
print("Computation time:%.2f,SNR:%.2fdB"%(compute_time,snr))
save_dir = '%s/%s_%s_Prediction_data.mat'%(args.save_dir,data_filename[0:-4],'SR0.5Mask5.mat')
sio.savemat(save_dir,{'Prediction_data':Prediction_data,'Original_data':Original_data,'Missing_data':Missing_data,'SNR':snr,'compute_time':compute_time})








