from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import platform
import math
import numpy as np
import scipy.io as sio
import models
import datasetting

parser = ArgumentParser(description='POCS-Net')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of POCS-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gpu_list', type=str, default='1', help='gpu index')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
batch_size = args.batch_size
learning_rate = args.learning_rate
layer_num = args.layer_num
gpu_list = args.gpu_list
model_dir = "./%s/Model_2DPOCSNet_learning_lr_%.4f_layernum_%02d" % (args.model_dir, learning_rate,layer_num)

log_file_name = "./%s/Log_2DPOCSNet_learning_lr_%.4f_layernum_%02d.txt" % (args.log_dir, learning_rate,layer_num)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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
    print('The modle is training based on GPU:%s'%gpu_list)
else:
    print('The modle is training based on CPU')
print('--------------------------------------Model Loading...--------------------------------------')
model = models.POCSNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)
print("Total number of param in 2DPOCSNet is ", sum(x.numel() for x in model.parameters()))

print('------------------------------------Data Set Loading...-------------------------------------')
filename = 'TrainingDataSet.h5'
filepath = '%s/%s'%(args.data_dir,filename)

features,labels,ntrain = datasetting.dataload(filepath,'Features','Labels')
print('The number of training pairs:%d'%ntrain)

if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=datasetting.RandomDataset(features, labels, ntrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=datasetting.RandomDataset(features , labels, ntrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

Loss_epoch = []
if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))
    filepath = "./%s/Loss_training_epoch_%d_lr_%.4f_layernum_%02d.mat"%(args.log_dir,start_epoch,learning_rate,layer_num)
    Loss_epoch = sio.loadmat(filepath)
    Loss_epoch = Loss_epoch["Loss_epoch_training"]
    Loss_epoch = Loss_epoch.tolist()[0]

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('--------------------------------------Model Training...--------------------------------------')
for epoch_i in range(start_epoch+1, end_epoch+1):
    Loss_all = 0
    for (batch_x,Phi) in rand_loader:
        batch_x = batch_x.to(device)
        Phi = Phi.to(device)
        Phi = Phi.view(-1,1,64,64)
        x = batch_x.view(-1,1,64,64)
        b = torch.mul(Phi,x)
        x0 = b
        [x_output, loss_layers_sym] = model(Phi,b,x0)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num-1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        Lambda = torch.Tensor([0.01]).to(device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(Lambda, loss_constraint)
        Loss_all = Loss_all + loss_all.item()
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Constraint Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(), loss_constraint)
        print(output_data)
    Loss_epoch.append(Loss_all/ math.ceil(ntrain/batch_size))
    Loss_epoch_np = np.array(Loss_epoch)
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 1 == 0:
        sio.savemat("./%s/Loss_training_epoch_%d_lr_%.4f_layernum_%02d.mat"%(args.log_dir,epoch_i,learning_rate,layer_num),{"Loss_epoch_training":Loss_epoch})
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters




