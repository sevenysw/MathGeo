from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import platform
import models
import datasetting
import numpy as np
import scipy.io as sio
import math
import utils
parser = ArgumentParser(description='POCS-Net')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start validation')
parser.add_argument('--end_epoch', type=int, default=50, help='epoch number of end validation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of POCS-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gpu_list', type=str, default='0,1', help='gpu index')
parser.add_argument('--model_dir', type=str, default='model_pre&post', help='trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='validation data directory')
parser.add_argument('--log_dir', type=str, default='SNR_training_post_3d2', help='log directory')
args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
batch_size = args.batch_size
learning_rate = args.learning_rate
layer_num = args.layer_num
gpu_list = args.gpu_list
model_dir = "./%s/Model_3DPOCSNet_learning_lr_%.4f_layernum_%02d" % (args.model_dir, learning_rate,layer_num)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
print('------------------------------------2D POCS-Net Setting-------------------------------------')
print('Start epoch:%d'%start_epoch)
print('End epoch:%d'%end_epoch)
print('Batch size:%d'%batch_size)
print('Learning rate:%.4f'%learning_rate)
print('Model directory:%s'%args.model_dir)
print('Log directory:%s'%args.log_dir)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('The modle is predicting based on GPU:%s'%gpu_list)
else:
    print('The modle is predicting based on CPU')
print('--------------------------------------Model Loading...--------------------------------------')
model = models.POCSNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)
print("Total number of param in 2DPOCSNet is ", sum(x.numel() for x in model.parameters()))

print('------------------------------------Data Set Loading...-------------------------------------')
filename = 'TrainingDataSet1.h5.h5'
filepath = '%s/%s'%(args.data_dir,filename)

features,labels,ntrain = datasetting.dataload(filepath,'Features','Labels')
print('The number of validation pairs:%d'%ntrain)

if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=datasetting.RandomDataset(features, labels, ntrain), batch_size=batch_size, num_workers=0,
                             shuffle=False)
else:
    rand_loader = DataLoader(dataset=datasetting.RandomDataset(features , labels, ntrain), batch_size=batch_size, num_workers=4,
                             shuffle=False)

print('-------------------------------------Model Validation...------------------------------------')

if start_epoch > 0:
    Loss_epoch = sio.loadmat("./%s/Loss_validation%d_lr_%.4f_layernum_%02d.mat"%(args.log_dir,start_epoch,learning_rate,layer_num))
    Loss_epoch = Loss_epoch["Loss_epoch_validation"]
    Loss_epoch = Loss_epoch.tolist()[0]
else:
    Loss_epoch = []

SNR_list = []
with torch.no_grad():
    for epoch_i in range(start_epoch+1, end_epoch+1):
        model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_i)))
        snr_sum = 0.
        count = 0
        for (batch_x,Phi) in rand_loader:
            batch_x = batch_x.to(device)
            Phi = Phi.to(device)
            Phi = Phi.view(-1,1,32,32,32)
            x = batch_x.view(-1,1,32,32,32)
            b = torch.mul(Phi,x)
            x0 = b
            [x_output, loss_layers_sym] = model(Phi,b,x0)
            x_output = x_output.view(-1,1,32,32,32)
            for i in range(x.shape[0]):
                original_slice = x[i,:,:,:,:]
                original_slice = original_slice.view(32,32,32)
                original_slice = original_slice.cpu().numpy()
                prediction_slice = x_output[i,:,:,:,:]
                prediction_slice = prediction_slice.view(32,32,32)
                prediction_slice = prediction_slice.cpu().numpy()
                snr = utils.SNR(original_slice,prediction_slice)
                if not math.isnan(snr) and not math.isinf(snr):
                    snr_sum = snr_sum + snr
                    count = count + 1
            output_data = "[%02d/%02d] Average SNR: %.2f dB.\n" % (epoch_i, end_epoch, snr_sum /count)
            print(output_data)
        SNR_list.append(snr_sum/count)
        SNR_list_np = np.array(SNR_list)
        sio.savemat("./%s/SNR_validation%d_lr_%.4f_layernum_%02d.mat"%(args.log_dir,epoch_i,learning_rate,layer_num),{"SNR_epoch_training":SNR_list})






